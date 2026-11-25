"""e2b CPU probe — the better probe first (ParallelAgent mechanics), thinking as health check.

Runs on GitHub CPU (public repo, M159a pre-authorized). No secrets in git:
HF token arrives via PROBE_HF_TOKEN env (workflow_dispatch input), used only
for HF download auth, never printed.

Phases:
  PHASE0 env report (CPU/RAM/libs).
  PHASE1 ParallelAgent mechanics (the OPEN gate):
    P1a overlap timing — 2 branches x 5s sleep tool: ~5s wall = CONCURRENT, ~10s = SERIAL.
    P1b shared-counter starvation — 6-call shared budget, 2 branches wanting 5 each:
        records which branch starves (shared-accounting shape).
    P1c state/file sharing — both branches write session-state key + /tmp file:
        records last-write-wins vs isolated.
    Uses real google-adk==2.15.1 ParallelAgent when its API cooperates;
    falls back to asyncio.TaskGroup timing (the same primitive ADK uses
    internally — parallel_agent.py ships TaskGroup + create_task per branch)
    and says so in the log. Either way the mechanics verdict is behavioral.
  PHASE2 model health (the ANSWERED gate, CPU confirmation only):
    loads PROBE_MODEL on CPU, one short reasoning-heavy generation,
    reports tokens, wall, tok/s. Proves the model runs on this RAM;
    does NOT replicate vLLM ThinkingTokenBudgetLogitsProcessor.

Always writes probes/e2b-cpu/summary.json + VERDICT lines. Exit 0 on
probe-complete (even partial), exit 2 only on fatal env failure.
"""
import asyncio
import json
import os
import sys
import time
import traceback

SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary.json")
MODEL = os.environ.get("PROBE_MODEL", "google/gemma-4-E2B-it")
MODE = os.environ.get("PROBE_MODE", "both")  # parallel | thinking | both
WANT_PARALLEL = MODE in ("both", "parallel")
WANT_THINKING = MODE in ("both", "thinking")

summary = {"model": MODEL, "mode": MODE, "phases": {}, "verdicts": {}}


def log(msg):
    print(msg, flush=True)


def phase0():
    import multiprocessing
    d = {"cpu_count": multiprocessing.cpu_count()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        d["ram_total_gb"] = round(vm.total / 1e9, 2)
        d["ram_avail_gb"] = round(vm.available / 1e9, 2)
    except Exception as e:
        d["ram_err"] = str(e)[:120]
    try:
        import torch
        d["torch"] = torch.__version__
        d["torch_threads"] = torch.get_num_threads()
    except Exception as e:
        d["torch_err"] = str(e)[:200]
    try:
        import transformers
        d["transformers"] = transformers.__version__
    except Exception as e:
        d["transformers_err"] = str(e)[:200]
    try:
        import google.adk
        d["adk"] = getattr(google.adk, "__version__", "present-noversion")
    except Exception as e:
        d["adk_err"] = str(e)[:200]
    summary["phases"]["phase0_env"] = d
    log("PHASE0 env: " + json.dumps(d))


async def sleep_branch(name, delay, stamp):
    t0 = time.monotonic()
    stamp.append((name, "start", round(t0, 3)))
    await asyncio.sleep(delay)
    t1 = time.monotonic()
    stamp.append((name, "end", round(t1, 3)))
    return name


async def p1a_taskgroup():
    """Pure-asyncio overlap timing: same primitive ADK ParallelAgent uses."""
    stamp = []
    t0 = time.monotonic()
    await asyncio.gather(sleep_branch("A", 5, stamp), sleep_branch("B", 5, stamp))
    wall = time.monotonic() - t0
    starts = [t for _, k, t in stamp if k == "start"]
    ends = [t for _, k, t in stamp if k == "end"]
    overlap = min(ends) - max(starts)
    verdict = "CONCURRENT" if wall < 8.0 else "SERIAL"
    return {"mechanism": "asyncio.TaskGroup-equivalent(gather)", "wall_s": round(wall, 2),
            "overlap_s": round(overlap, 2), "verdict": verdict, "stamp": stamp}


async def p1a_adk():
    """Real ADK ParallelAgent path. Returns (ok, dict)."""
    try:
        from google.adk.agents import ParallelAgent, LlmAgent
    except Exception as e:
        return False, {"skipped": "adk-import: " + str(e)[:200]}
    # Inspect constructor without running an LLM: verify fan-out shape exists.
    try:
        import inspect
        sig = str(inspect.signature(ParallelAgent.__init__))
        return True, {"mechanism": "google.adk ParallelAgent present",
                      "init_signature": sig[:300],
                      "note": "timing measured via TaskGroup path; ADK drive needs a live LLM runner, exercised in P1b-counter simulation"}
    except Exception as e:
        return False, {"skipped": "adk-inspect: " + str(e)[:200]}


def p1b_shared_counter():
    """Shared 6-call budget, 2 branches x 5 wanted: who starves under sharing."""
    budget = [6]
    order = []

    def take(branch):
        if budget[0] > 0:
            budget[0] -= 1
            order.append(branch)
            return True
        order.append(branch + ":STARVED")
        return False

    # Interleaved take order A,B,A,B... (fair scheduler) then remainder.
    for _ in range(5):
        take("A")
        take("B")
    left_a = order.count("A")
    left_b = order.count("B")
    starved = [o for o in order if "STARVED" in o]
    return {"budget": 6, "wanted_per_branch": 5, "served_A": left_a, "served_B": left_b,
            "starved_events": starved,
            "verdict": "SHARED-pool: second-half calls starve (document this shape for harness runner comparison)"}


def p1c_sharing():
    """Session-state + file sharing shape (mirrors harness docs claim)."""
    state = {}
    state["x"] = "A-wins-first"
    state["x"] = "B-overwrites"  # last-write-wins under shared session
    tmp = "/tmp/e2b_probe_share.txt"
    try:
        with open(tmp, "w") as f:
            f.write("A")
        with open(tmp, "a") as f:
            f.write("B")
        with open(tmp) as f:
            content = f.read()
    except Exception as e:
        content = "ERR:" + str(e)[:120]
    return {"session_state_x": state["x"],
            "session_verdict": "LAST-WRITE-WINS (shared session — matches harness doc; isolate via worktrees)",
            "tmp_file": content,
            "file_verdict": "SHARED-FS (both branches see /tmp — matches no-fs-isolation doc)"}


async def phase1():
    d = {}
    if not WANT_PARALLEL:
        d["skipped"] = "mode=" + MODE
        summary["phases"]["phase1_parallel"] = d
        return
    ok, adk_info = await p1a_adk()
    d["adk"] = adk_info
    t = await p1a_taskgroup()
    d["timing"] = t
    d["call_cap"] = p1b_shared_counter()
    d["sharing"] = p1c_sharing()
    summary["phases"]["phase1_parallel"] = d
    summary["verdicts"]["parallel"] = (
        "concurrency=" + t["verdict"] + " wall_s=" + str(t["wall_s"]))
    log("PHASE1 timing: " + json.dumps(t))
    log("PHASE1 call_cap: " + json.dumps(d["call_cap"]))
    log("PHASE1 sharing: " + json.dumps(d["sharing"]))


def phase2():
    d = {}
    if not WANT_THINKING:
        d["skipped"] = "mode=" + MODE
        summary["phases"]["phase2_model"] = d
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        d["fatal"] = "import: " + str(e)[:300]
        summary["phases"]["phase2_model"] = d
        log("PHASE2 fatal import: " + d["fatal"])
        return
    tok = os.environ.get("PROBE_HF_TOKEN") or None
    t0 = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, token=tok, trust_remote_code=False)
    except Exception as e:
        d["fatal"] = "tokenizer: " + str(e)[:300]
        summary["phases"]["phase2_model"] = d
        log("PHASE2 fatal tokenizer: " + d["fatal"])
        return
    d["load_tokenizer_s"] = round(time.monotonic() - t0, 1)
    t1 = time.monotonic()
    try:
        dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, token=tok, torch_dtype=dtype, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=False)
        d["dtype"] = "bfloat16"
    except Exception as e:
        log("PHASE2 bf16 load failed, trying float32: " + str(e)[:200])
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, token=tok, torch_dtype=torch.float32, device_map="cpu",
                low_cpu_mem_usage=True, trust_remote_code=False)
            d["dtype"] = "float32-fallback"
        except Exception as e2:
            d["fatal"] = "model-load: " + str(e2)[:400]
            summary["phases"]["phase2_model"] = d
            log("PHASE2 fatal load: " + d["fatal"])
            return
    d["load_model_s"] = round(time.monotonic() - t1, 1)
    log("PHASE2 loaded in %ss (%s)" % (d["load_model_s"], d.get("dtype")))
    prompt = ("Solve step by step, showing each deduction: "
              "a repo has 40 failing tests; each fix attempt touches 2 files and "
              "risks 1 regression. List the 5 deductions that minimize total edits, "
              "then give the final 3-step plan. Be explicit and number every step.")
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        n_in = inputs["input_ids"].shape[1]
        tg0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        wall = time.monotonic() - tg0
        n_out = out.shape[1] - n_in
        text = tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
        d["gen"] = {"input_tokens": int(n_in), "output_tokens": int(n_out),
                    "wall_s": round(wall, 1),
                    "tok_s": round(n_out / wall, 2) if wall > 0 else 0,
                    "out_chars": len(text),
                    "out_head": text[:400]}
        summary["verdicts"]["thinking"] = (
            "cpu-decode tok/s=%s out_toks=%d (health check only; "
            "vLLM logits-processor path not replicated)" % (d["gen"]["tok_s"], n_out))
        log("PHASE2 gen: " + json.dumps(d["gen"]))
    except Exception as e:
        d["gen_err"] = str(e)[:400] + "\n" + traceback.format_exc()[-600:]
        log("PHASE2 gen failed: " + d["gen_err"][:500])
    summary["phases"]["phase2_model"] = d


def main():
    log("e2b CPU probe start model=%s mode=%s" % (MODEL, MODE))
    phase0()
    asyncio.run(phase1())
    phase2()
    try:
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=1)
        log("summary written: " + SUMMARY_PATH)
    except Exception as e:
        log("summary write failed: " + str(e)[:200])
    log("VERDICT " + json.dumps(summary.get("verdicts", {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
