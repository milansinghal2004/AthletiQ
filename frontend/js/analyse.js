/* =============================================
   ANALYSE — Triggers pose_estimation.py via backend
   ============================================= */

export function initAnalyse() {
  const btn = document.getElementById('analyseBtn');
  const statusEl = document.getElementById('analyseStatus');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'RUNNING...';
    statusEl.style.display = 'flex';
    statusEl.innerHTML = '<span style="color:var(--acid2)">⟳ Running pose_estimation.py...</span>';

    try {
      const res = await fetch('/run-analysis', { method: 'POST' });
      const data = await res.json();

      if (data.success) {
        statusEl.innerHTML = '<span style="color:var(--acid)">✓ Analysis complete — ' + (data.message || 'Done') + '</span>';
        btn.textContent = 'ANALYSE SHOT';
      } else {
        statusEl.innerHTML = '<span style="color:var(--fire)">✗ Error: ' + (data.error || 'Unknown error') + '</span>';
        btn.textContent = 'ANALYSE SHOT';
      }
    } catch (err) {
      statusEl.innerHTML = '<span style="color:var(--fire)">✗ Could not reach backend. Is the server running?</span>';
      btn.textContent = 'ANALYSE SHOT';
    }

    btn.disabled = false;
  });
}
