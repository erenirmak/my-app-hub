document.addEventListener('DOMContentLoaded', function() {
    const header = document.querySelector('.sliding-header');
    header.classList.add('active');

    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;

            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
    const collapsibleContents = document.querySelectorAll('.collapsible-content');

    collapsibleHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const targetId = header.dataset.target;
            const content = document.getElementById(targetId);

            // Close all other collapsible contents
            collapsibleContents.forEach(otherContent => {
                if (otherContent !== content) {
                    otherContent.classList.remove('active');
                }
            });

            // Toggle the current content
            content.classList.toggle('active');

            // Activate the first tab if the content is active
            if (content.classList.contains('active')) {
                activateFirstTab(targetId);
            }
        });
    });
});

function openLightbox(imageUrl) {
    document.getElementById('lightbox-image').src = imageUrl;
    document.getElementById('lightbox').style.display = 'block';
}

document.getElementById('lightbox').onclick = function() {
    document.getElementById('lightbox').style.display = 'none';
};

function activateFirstTab(collapsibleId) {
    const tabButtons = document.querySelectorAll(`#${collapsibleId} .tab-button`);
    const tabContents = document.querySelectorAll(`#${collapsibleId} .tab-content`);

    if (tabButtons.length > 0 && tabContents.length > 0) {
        // Remove active class from all tabs
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));

        // Activate the first tab
        tabButtons[0].classList.add('active');
        tabContents[0].classList.add('active');
    }
}