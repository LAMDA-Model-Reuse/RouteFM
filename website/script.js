// One player per paper: language changes never leave a second narrator playing.
document.querySelectorAll("[data-paper-video]").forEach((showcase) => {
  const video = showcase.querySelector("video");
  const source = video.querySelector("source");
  const track = video.querySelector("track");
  const slug = showcase.dataset.paperVideo;
  const buttons = showcase.querySelectorAll("[data-video-language]");
  let selectedLanguage = "en";
  const selectLanguage = (language, updateUrl = true) => {
    if (!["en", "zh"].includes(language)) return;
    buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.videoLanguage === language)));
    if (language !== selectedLanguage) {
      video.pause();
      source.src = "videos/" + slug + "-" + language + ".mp4";
      track.src = "videos/" + slug + "-" + language + ".vtt";
      track.srclang = language;
      track.label = language === "en" ? "English" : "中文";
      video.poster = "assets/" + slug + "-" + language + "-poster.png";
      video.setAttribute("aria-label", slug + " " + (language === "en" ? "English" : "Chinese") + " paper overview");
      video.load();
      selectedLanguage = language;
    }
    showcase.querySelectorAll("[data-video-download]").forEach((link) => {
      link.href = "videos/" + slug + "-" + language + "." + link.dataset.videoDownload;
    });
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set("video", language);
      url.hash = "video";
      history.replaceState(null, "", url);
    }
  };
  buttons.forEach((button) => button.addEventListener("click", () => selectLanguage(button.dataset.videoLanguage)));
  selectLanguage(new URLSearchParams(window.location.search).get("video") || "en", false);
});

const copyButton = document.querySelector("[data-copy-target]");

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const target = document.getElementById(copyButton.dataset.copyTarget);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.innerText.trim());
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => { copyButton.textContent = original; }, 1600);
    } catch {
      copyButton.textContent = "Select to copy";
    }
  });
}

document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", () => {
    const target = document.querySelector(link.getAttribute("href"));
    if (target) target.setAttribute("tabindex", "-1");
  });
});
