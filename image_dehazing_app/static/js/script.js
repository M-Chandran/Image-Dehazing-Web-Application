document.addEventListener('DOMContentLoaded', function() {
    console.log('DeHaze AI - Modern UI Loaded');

    // Mobile menu toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    // File upload handling
    const fileInput = document.getElementById('file');
    const uploadForm = document.getElementById('upload-form');
    const dehazeBtn = document.getElementById('dehaze-btn');

    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const fileName = file.name;
                
                // Update UI to show selected file
                const fileInputLabel = document.querySelector('.file-input-label');
                if (fileInputLabel) {
                    const textElement = fileInputLabel.querySelector('.file-input-text h3');
                    if (textElement) {
                        textElement.innerHTML = `Selected: <strong>${fileName}</strong>`;
                    }
                    fileInputLabel.style.borderColor = 'var(--primary-teal)';
                }
                
                // Validate file type
                const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg'];
                if (!allowedTypes.includes(file.type)) {
                    alert('Please select a valid image file (PNG, JPG, JPEG).');
                    fileInput.value = '';
                    return;
                }
                
                // Validate file size (50MB max)
                if (file.size > 50 * 1024 * 1024) {
                    alert('File size must be less than 50MB.');
                    fileInput.value = '';
                    return;
                }
            }
        });

        // Drag and drop support
        const fileInputWrapper = document.querySelector('.file-input-wrapper');
        if (fileInputWrapper) {
            fileInputWrapper.addEventListener('dragover', function(e) {
                e.preventDefault();
                fileInputWrapper.querySelector('.file-input-label').classList.add('dragover');
            });

            fileInputWrapper.addEventListener('dragleave', function(e) {
                e.preventDefault();
                fileInputWrapper.querySelector('.file-input-label').classList.remove('dragover');
            });

            fileInputWrapper.addEventListener('drop', function(e) {
                e.preventDefault();
                fileInputWrapper.querySelector('.file-input-label').classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    // Trigger change event
                    const event = new Event('change');
                    fileInput.dispatchEvent(event);
                }
            });
        }
    }

    // Form submission
    if (uploadForm && dehazeBtn) {
        uploadForm.addEventListener('submit', function(e) {
            if (!fileInput || !fileInput.files.length) {
                alert('Please select a file first.');
                e.preventDefault();
                return;
            }

            // Show loading state
            const originalHTML = dehazeBtn.innerHTML;
            dehazeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
            dehazeBtn.disabled = true;

            // Re-enable after 30 seconds as fallback
            setTimeout(() => {
                dehazeBtn.innerHTML = originalHTML;
                dehazeBtn.disabled = false;
            }, 30000);
        });
    }

    // Alert close buttons
    const alertCloseButtons = document.querySelectorAll('.alert .alert-close, .alert .btn-close');
    alertCloseButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alert = this.closest('.alert');
            if (alert) {
                alert.style.opacity = '0';
                setTimeout(() => {
                    alert.remove();
                }, 300);
            }
        });
    });

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                alert.remove();
            }, 300);
        }, 5000);
    });

    // Scroll reveal animations
    const scrollElements = document.querySelectorAll('.scroll-reveal');
    if (scrollElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1
        });

        scrollElements.forEach(el => observer.observe(el));
    }

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});
