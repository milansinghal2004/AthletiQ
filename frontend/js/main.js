/* =============================================
   MAIN — Entry point, imports all modules
   ============================================= */
import { initCursor } from './cursor.js';
import { initScrollReveal, initCounters, initSmoothScroll } from './animations.js';
import { initAnalyse } from './analyse.js';

document.addEventListener('DOMContentLoaded', () => {
  initCursor();
  initScrollReveal();
  initCounters();
  initSmoothScroll();
  initAnalyse();
});
