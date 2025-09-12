(function () {
  const parent = window.parent;

  // ⚠️ Si une page a demandé le scroll, on ne lock pas.
  if (parent.__ALLOW_SCROLL__) return;

  // Déjà installé ? ne pas ré-attacher
  if (parent.__SCROLL_LOCK__) { parent.__SCROLL_LOCK__.enable(); return; }

  function isInsideSelect(target) {
    const doc = parent.document;
    const t = target || doc.activeElement;
    return (
      t?.closest('div[data-baseweb="select"]') ||
      t?.closest('[role="combobox"]') ||
      t?.closest('[role="listbox"]') ||
      (doc.activeElement && (
        doc.activeElement.closest('div[data-baseweb="select"]') ||
        ['combobox','listbox'].includes(doc.activeElement.getAttribute('role') || '')
      ))
    );
  }

  const api = {
    enabled: true,
    attached: false,
    mo: null,
    stopScroll: null,
    onKeyDown: null,
    applyStyles() {
      const app   = parent.document.querySelector('[data-testid="stAppViewContainer"]');
      const main  = parent.document.querySelector('[data-testid="stMain"]');
      const block = parent.document.querySelector('.block-container');
      [app, main, block].forEach(el => {
        if (!el) return;
        el.style.overflow = 'hidden';
        el.style.height   = '100vh';
      });
    },
    resetStyles() {
      const app   = parent.document.querySelector('[data-testid="stAppViewContainer"]');
      const main  = parent.document.querySelector('[data-testid="stMain"]');
      const block = parent.document.querySelector('.block-container');
      [app, main, block].forEach(el => {
        if (!el) return;
        el.style.overflow = '';
        el.style.height   = '';
      });
    },
    attach() {
      if (api.attached || !api.enabled) return;

      api.stopScroll = function(e){
        if (
          e.target.closest('form[role="form"]') ||
          e.target.closest('[role="dialog"]') ||
          e.target.closest('.ag-root') ||
          isInsideSelect(e.target)
        ) return;
        e.preventDefault();
        e.stopPropagation();
        return false;
      };
      api.onKeyDown = function(e){
        const keys = ['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' '];
        if (!keys.includes(e.key)) return;
        if (isInsideSelect(e.target)) return;
        e.preventDefault();
        e.stopPropagation();
      };

      parent.addEventListener('wheel',     api.stopScroll, { passive:false });
      parent.addEventListener('touchmove', api.stopScroll, { passive:false });
      parent.addEventListener('keydown',   api.onKeyDown,  true);

      api.applyStyles();

      api.mo = new parent.MutationObserver(() => { if (api.enabled) api.applyStyles(); });
      api.mo.observe(parent.document.body, { childList: true, subtree: true });

      api.attached = true;
    },
    detach() {
      if (!api.attached) return;
      try {
        parent.removeEventListener('wheel',     api.stopScroll, { passive:false });
        parent.removeEventListener('touchmove', api.stopScroll, { passive:false });
        parent.removeEventListener('keydown',   api.onKeyDown,  true);
      } catch(e) {}
      if (api.mo) { try { api.mo.disconnect(); } catch(e) {} api.mo = null; }
      api.resetStyles();
      api.attached = false;
    },
    enable()  { api.enabled = true;  api.attach();  },
    disable() { api.enabled = false; api.detach();  }
  };

  parent.__SCROLL_LOCK__ = api;
  api.enable();  // comportement par défaut = lock activé
})();