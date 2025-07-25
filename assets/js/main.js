document.addEventListener('DOMContentLoaded', () => {

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

  // PDF Download functionality
  const downloadPdfBtn = document.getElementById('downloadPdfBtn');
  if (downloadPdfBtn) {
    downloadPdfBtn.addEventListener('click', () => {
      // Get the main content area (excluding the button itself)
      const content = document.querySelector('.container');
      
      // Configure PDF options
      const opt = {
        margin: [10, 10, 10, 10],
        filename: 'Aditya_Bharadwaj_Resume.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { 
          scale: 2,
          useCORS: true,
          letterRendering: true
        },
        jsPDF: { 
          unit: 'mm', 
          format: 'a4', 
          orientation: 'portrait' 
        }
      };

      // Generate and download the PDF
      html2pdf().set(opt).from(content).save();
    });
  }

}); 