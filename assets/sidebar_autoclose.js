window.addEventListener("load", function () {
  const sidebar = window.parent.document.querySelector("section[data-testid='stSidebar']");
  let timeoutID = null;

  if (!sidebar) return;

  sidebar.addEventListener("mouseenter", function () {
    if (timeoutID) { clearTimeout(timeoutID); timeoutID = null; }
  });

  sidebar.addEventListener("mouseleave", function () {
    timeoutID = setTimeout(function () {
      const toggle = window.parent.document.querySelector("span[data-testid='stIconMaterial']");
      if (toggle && toggle.textContent.includes("keyboard_double_arrow_left")) {
        toggle.click();
      }
    }, 300);
  });
});
