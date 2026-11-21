// scramble.mjs — public-runner encryption for mass-spec-cpu (adapted from bot-paper pattern).
// Actions encrypts results/*.json with AES-256-GCM (SCRAMBLE_KEY secret) before push.
// We decrypt locally after `git pull` using the key from the vault.
// Wrapper: {enc:"AES-256-GCM-v1", nonce, data, tag, sha256, bytes, date}.
import {readFileSync, writeFileSync} from "fs";
import {createCipheriv, createDecipheriv, createHash, randomBytes} from "crypto";

const ALG = "AES-256-GCM-v1";

function loadKeyHex() {
  const h = (process.env.SCRAMBLE_KEY || "").trim();
  if (/^[0-9a-fA-F]{64}$/.test(h)) return h.toLowerCase();
  return null;
}

function loadKeyFromFile(path) {
  const raw = readFileSync(path, "utf8").trim();
  try {
    const j = JSON.parse(raw);
    if (j && typeof j.keyHex === "string" && /^[0-9a-fA-F]{64}$/.test(j.keyHex.trim()))
      return j.keyHex.trim().toLowerCase();
  } catch { /* not JSON — treat as raw */ }
  if (/^[0-9a-fA-F]{64}$/.test(raw)) return raw.toLowerCase();
  throw new Error("key file must hold 64 hex chars (or JSON with keyHex)");
}

function sha256hex(buf) { return createHash("sha256").update(buf).digest("hex"); }

function encryptBuffer(plain, keyHex) {
  const key = Buffer.from(keyHex, "hex");
  const nonce = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key, nonce);
  const ct = Buffer.concat([cipher.update(plain), cipher.final()]);
  return {
    enc: ALG, nonce: nonce.toString("base64"), data: ct.toString("base64"),
    tag: cipher.getAuthTag().toString("base64"),
    sha256: sha256hex(plain), bytes: plain.length,
    date: new Date().toISOString().slice(0, 10),
  };
}

function decryptWrapper(w, keyHex) {
  if (!w || w.enc !== ALG) throw new Error("not a scramble wrapper (missing enc field)");
  const d = createDecipheriv("aes-256-gcm", Buffer.from(keyHex, "hex"), Buffer.from(w.nonce, "base64"));
  d.setAuthTag(Buffer.from(w.tag, "base64"));
  const plain = Buffer.concat([d.update(Buffer.from(w.data, "base64")), d.final()]);
  if (w.sha256 && sha256hex(plain) !== w.sha256) throw new Error("sha256 mismatch after decrypt");
  return plain;
}

function isWrapper(o) { return o && o.enc === ALG && typeof o.data === "string" && typeof o.nonce === "string"; }

const [, , cmd, ...rest] = process.argv;
if (cmd === "encrypt") {
  const keyHex = loadKeyHex();
  if (!keyHex) { console.error("missing/invalid SCRAMBLE_KEY (need 64 hex chars in env)"); process.exit(1); }
  for (const f of rest.filter(a => !a.startsWith("--"))) {
    let raw;
    try { raw = readFileSync(f); } catch (e) { console.error("skip " + f + ": cannot read"); process.exit(1); }
    try { if (isWrapper(JSON.parse(raw.toString("utf8")))) { console.log("already scrambled: " + f); continue; } } catch { /* raw */ }
    const w = encryptBuffer(raw, keyHex);
    writeFileSync(f, JSON.stringify(w, null, 1) + "\n");
    console.log("scrambled " + f + " bytes=" + w.bytes + " sha256=" + w.sha256.slice(0, 16) + "…");
  }
} else if (cmd === "decrypt") {
  let keyFile = null, out = null;
  const files = [];
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === "--key-file" && rest[i + 1]) keyFile = rest[++i];
    else if (rest[i] === "--out" && rest[i + 1]) out = rest[++i];
    else if (!rest[i].startsWith("--")) files.push(rest[i]);
  }
  const keyHex = keyFile ? loadKeyFromFile(keyFile) : loadKeyHex();
  if (!keyHex) { console.error("no key: pass --key-file <path> or set SCRAMBLE_KEY env"); process.exit(1); }
  for (const f of files) {
    const plain = decryptWrapper(JSON.parse(readFileSync(f, "utf8")), keyHex);
    const target = out || (f.endsWith(".json") ? f.replace(/\.json$/, ".dec.json") : f + ".dec");
    writeFileSync(target, plain);
    console.log("decrypted " + f + " -> " + target + " bytes=" + plain.length);
  }
} else {
  console.error("usage: scramble.mjs encrypt <file...> | decrypt <file...> [--key-file k] [--out o]");
  process.exit(1);
}
