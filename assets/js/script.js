document.addEventListener('DOMContentLoaded', () => {
    // Navigation buttons
    const navButtons = document.querySelectorAll('.nav-links a, .btn-primary, .btn-secondary');
    navButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const href = button.getAttribute('href');
            if (href) {
                window.location.href = href;
            }
        });
    });

    // Role selection on register page
    const roleSelect = document.querySelectorAll('.role-select .card');
    if (roleSelect) {
        roleSelect.forEach(card => {
            card.addEventListener('click', () => {
                roleSelect.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
            });
        });
    }
});