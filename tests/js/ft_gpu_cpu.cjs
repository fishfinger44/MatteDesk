// MatteDesk – Copyright (C) 2026 Lech Gruszewski
// SPDX-License-Identifier: GPL-3.0-or-later
// node tests/js/ft_gpu_cpu.cjs < job.json > out.json – the JS reference of the Final Touch GPU preview (static/ft_gpu.js
// cpu()) for tests/test_ft_gpu.py: job = {W, H, PW, PH, base: base64 u8 PW·PH, strokes, live, finish} -> {out: base64}
const path = require('path');
const ft = require(path.join(__dirname, '..', '..', 'static', 'ft_gpu.js'));
let buf = '';
process.stdin.on('data', (d) => { buf += d; });
process.stdin.on('end', () => {
  const jobs = JSON.parse(buf);
  const res = jobs.map((j) => {
    const base = new Uint8Array(Buffer.from(j.base, 'base64'));
    const t0 = process.hrtime.bigint();
    const out = ft.cpu({ ...j, base });
    return { out: Buffer.from(out).toString('base64'), ms: Number(process.hrtime.bigint() - t0) / 1e6 };
  });
  process.stdout.write(JSON.stringify(res));
});
