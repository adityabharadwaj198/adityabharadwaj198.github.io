document.addEventListener('DOMContentLoaded', () => {
  const themeToggle = document.querySelector('.theme-toggle');
  const savedTheme = localStorage.getItem('theme');
  const preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';

  document.documentElement.dataset.theme = savedTheme || preferredTheme;

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const nextTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = nextTheme;
      localStorage.setItem('theme', nextTheme);
    });
  }

  // Find all images with the 'enlargeable-image' class
  const enlargeableImages = document.querySelectorAll('.enlargeable-image');
  
  // Add a click event listener to each enlargeable image
  enlargeableImages.forEach(image => {
    image.addEventListener('click', () => {
      // Create and show the lightbox
      basicLightbox.create(`
        <img src="${image.src}" style="display: block; max-width: 90vw; max-height: 90vh; margin: auto;">
      `).show();
    });
  });

}); 