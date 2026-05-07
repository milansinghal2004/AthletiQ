/* =============================================
   CURSOR — Custom cursor + bat emoji toggle
   ============================================= */

export function initCursor() {
  const cursor = document.getElementById('cursor');
  const ring = document.getElementById('cursorRing');
  const batCursor = document.getElementById('batCursor');
  const toggleBtn = document.getElementById('cursorToggleBtn');
  let mx = 0, my = 0, rx = 0, ry = 0;
  let batMode = false;

  document.addEventListener('mousemove', e => {
    mx = e.clientX;
    my = e.clientY;
  });

  function animCursor() {
    if (!batMode) {
      cursor.style.left = mx + 'px';
      cursor.style.top = my + 'px';
      rx += (mx - rx) * 0.15;
      ry += (my - ry) * 0.15;
      ring.style.left = rx + 'px';
      ring.style.top = ry + 'px';
    } else {
      batCursor.style.left = mx + 'px';
      batCursor.style.top = my + 'px';
    }
    requestAnimationFrame(animCursor);
  }
  animCursor();

  // Toggle between normal and bat cursor
  toggleBtn.addEventListener('click', () => {
    batMode = !batMode;
    if (batMode) {
      cursor.style.opacity = '0';
      ring.style.opacity = '0';
      batCursor.style.opacity = '1';
      document.body.style.cursor = 'none';
      toggleBtn.textContent = '🔵 Normal Cursor';
    } else {
      cursor.style.opacity = '1';
      ring.style.opacity = '1';
      batCursor.style.opacity = '0';
      toggleBtn.textContent = '🏏 Bat Cursor';
    }
  });

  const hoverTargets = 'a, button, .pipe-step, .feat-card, .upload-zone';
  document.querySelectorAll(hoverTargets).forEach(el => {
    el.addEventListener('mouseenter', () => {
      if (batMode) { batCursor.style.fontSize = '32px'; return; }
      cursor.style.width = '14px';
      cursor.style.height = '14px';
      ring.style.width = '56px';
      ring.style.height = '56px';
      ring.style.borderColor = 'rgba(0,255,136,0.8)';
    });
    el.addEventListener('mouseleave', () => {
      if (batMode) { batCursor.style.fontSize = '24px'; return; }
      cursor.style.width = '8px';
      cursor.style.height = '8px';
      ring.style.width = '36px';
      ring.style.height = '36px';
      ring.style.borderColor = 'rgba(0,255,136,0.5)';
    });
  });
}
