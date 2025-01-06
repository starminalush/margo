const canvas = document.getElementById('pdf-canvas');
const context = canvas.getContext('2d');
const chatInput = document.getElementById('chat-input');
const chatOutput = document.getElementById('chat-output');
const askButton = document.getElementById('ask-btn');
const pdfUpload = document.getElementById('pdf-upload');
const prevPageBtn = document.getElementById('prev-page');
const nextPageBtn = document.getElementById('next-page');
const pageNumDisplay = document.getElementById('page-num');
const pageCountDisplay = document.getElementById('page-count');

let pdfDoc = null;
let currentPage = 1;
let pageText = {}; // Для хранения текста каждой страницы
let highlights = []; // Для хранения выделений

let selecting = false; // Флаг, определяющий, в режиме ли мы выделения
let selectionStart = null; // Начало выделения
let selectionEnd = null; // Конец выделения
let pageViewport = null; // Для хранения масштаба текущей страницы

// Функция для отображения PDF на канвасе
function renderPDF(pdfFile) {
    const fileReader = new FileReader();
    fileReader.onload = function (e) {
        const typedArray = new Uint8Array(e.target.result);
        pdfjsLib.getDocument(typedArray).promise.then(function (pdf) {
            pdfDoc = pdf;
            renderPage(currentPage);  // Отображаем первую страницу PDF
            pageCountDisplay.textContent = pdfDoc.numPages;
            extractTextFromPage(currentPage); // Извлекаем текст с первой страницы
        });
    };
    fileReader.readAsArrayBuffer(pdfFile);
}

// Функция для рендеринга конкретной страницы PDF
function renderPage(pageNum) {
    pdfDoc.getPage(pageNum).then(function (page) {
        pageViewport = page.getViewport({ scale: 1 }); // Сохраняем информацию о масштабе
        canvas.width = pageViewport.width;
        canvas.height = pageViewport.height;

        // Отображаем PDF на канвасе (без очистки)
        page.render({
            canvasContext: context,
            viewport: pageViewport
        });

        pageNumDisplay.textContent = pageNum;

        // Рисуем выделения после рендеринга страницы
        highlights.forEach(highlight => {
            drawSelection(highlight.start, highlight.end);
        });
    });
}

// Извлечение текста с конкретной страницы
function extractTextFromPage(pageNum) {
    pdfDoc.getPage(pageNum).then(function (page) {
        page.getTextContent().then(function (textContent) {
            const textItems = textContent.items;
            pageText[pageNum] = textItems; // Сохраняем текст и координаты страницы
        });
    });
}

// Обработка загрузки PDF
pdfUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
        currentPage = 1;
        renderPDF(file);
    } else {
        alert('Пожалуйста, загрузите PDF файл.');
    }
});

// Переход на предыдущую страницу
prevPageBtn.addEventListener('click', function () {
    if (currentPage > 1) {
        currentPage--;
        renderPage(currentPage);
        extractTextFromPage(currentPage); // Извлекаем текст для новой страницы
    }
});

// Переход на следующую страницу
nextPageBtn.addEventListener('click', function () {
    if (currentPage < pdfDoc.numPages) {
        currentPage++;
        renderPage(currentPage);
        extractTextFromPage(currentPage); // Извлекаем текст для новой страницы
    }
});

// Функция для обработки выделения текста
canvas.addEventListener('mousedown', function (e) {
    selecting = true;
    selectionStart = { x: e.offsetX, y: e.offsetY }; // Запоминаем начало выделения
});

canvas.addEventListener('mousemove', function (e) {
    if (selecting && selectionStart) {
        selectionEnd = { x: e.offsetX, y: e.offsetY };

        // Рисуем только выделение поверх страницы
        drawSelection(selectionStart, selectionEnd);
    }
});

canvas.addEventListener('mouseup', function () {
    selecting = false;
    if (selectionStart && selectionEnd) {
        // Сохраняем выделение
        highlights.push({ start: selectionStart, end: selectionEnd });
        getSelectedText(selectionStart, selectionEnd);
    }
});

function drawSelection(start, end) {
    context.fillStyle = 'rgba(255, 255, 0, 0.5)';
    const width = end.x - start.x;
    const height = end.y - start.y;

    // Рисуем выделение
    context.fillRect(start.x, start.y, width, height);
}

function getSelectedText(start, end) {
    const pageTextContent = pageText[currentPage];
    const selectedText = [];

    // Масштабируем координаты выделенной области
    const scale = pageViewport.scale;

    // Переводим координаты выделенной области с учетом масштаба
    const startX = start.x / scale;
    const startY = start.y / scale;
    const endX = end.x / scale;
    const endY = end.y / scale;

    // Ищем текст в пределах выделенной области
    pageTextContent.forEach(function (item) {
        const x1 = item.transform[4]; // Координата x
        const y1 = item.transform[5]; // Координата y
        const width = item.width;
        const height = item.height;

        // Проверяем, попадает ли текст в выделенную область
        if (
            x1 + width >= startX && x1 <= endX &&
            y1 + height >= startY && y1 <= endY
        ) {
            selectedText.push(item.str); // Добавляем текст в выделенную область
        }
    });

    if (selectedText.length > 0) {
        const text = selectedText.join(' ');
        chatOutput.innerHTML += `<div><strong>Выделено:</strong> ${text}</div>`;
    }
}
