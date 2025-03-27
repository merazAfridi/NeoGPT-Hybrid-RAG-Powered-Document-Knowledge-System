async function uploadFiles() {
    const fileInput = document.getElementById('fileInput');
    const progress = document.getElementById('uploadProgress');
    const progressBar = progress.querySelector('.progress-bar');
    const status = document.getElementById('uploadStatus');

    if (!fileInput.files.length) {
        showStatus('error', 'Please select files to upload');
        return;
    }

    //reset and show progress
    progress.classList.remove('d-none');
    status.innerHTML = '';
    progressBar.style.width = '0%';

    try {
        const formData = new FormData();
        for (const file of fileInput.files) {
            formData.append('file', file);
        }

        showStatus('info', 'Uploading files...');
        progressBar.style.width = '50%';

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.errors) {
            const errorMessages = result.errors
                .map(err => `${err.filename}: ${err.error}`)
                .join('\n');
            showStatus('warning', `Some files failed to upload:\n${errorMessages}`);
        }

        if (result.processed && result.processed.length > 0) {
            progressBar.style.width = '100%';
            showStatus('success', `Successfully processed ${result.processed.length} file(s)`);
            
            //update documents list
            if (result.documents) {
                updateDocumentsList(result.documents);
            }

            // Show status for each file
            result.processed.forEach(file => {
                showFileStatus(file.filename, 'success');
            });
        }

        //reset after delay
        setTimeout(() => {
            progress.classList.add('d-none');
            fileInput.value = '';
        }, 2000);

    } catch (err) {
        showStatus('error', 'Upload failed: ' + err.message);
        progress.classList.add('d-none');
    }
}

async function queryDocuments() {
    const documentsSelect = document.getElementById('documents');
    const queryInput = document.getElementById('queryInput');
    const answer = document.getElementById('answer');
    
    const selectedDocs = Array.from(documentsSelect.selectedOptions).map(opt => opt.value);
    
    if (!selectedDocs.length) {
        showStatus('error', 'Please select at least one document');
        return;
    }

    if (!queryInput.value.trim()) {
        showStatus('error', 'Please enter a question');
        return;
    }

    try {
        answer.innerHTML = `
            <div class="d-flex align-items-center gap-2 text-primary">
                <div class="spinner-border spinner-border-sm"></div>
                <span>Processing your question...</span>
            </div>`;
        
        const response = await fetch("/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                documents: selectedDocs,
                query: queryInput.value.trim()
            })
        });

        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        answer.innerHTML = formatAnswer(result.answer);

    } catch (err) {
        answer.innerHTML = formatError(err.message);
    }
}

function updateProgress(current, total) {
    const progressBar = document.querySelector('.progress-bar');
    const percentage = (current / total) * 100;
    progressBar.style.width = `${percentage}%`;
}

function showFileStatus(filename, status, error = '') {
    const statusDiv = document.getElementById('uploadStatus');
    const icon = status === 'uploading' ? 'arrow-repeat spin' :
                status === 'success' ? 'check-circle' : 'x-circle';
    const color = status === 'uploading' ? 'text-primary' :
                 status === 'success' ? 'text-success' : 'text-danger';

    const html = `
        <div class="file-status fade-in">
            <i class="bi bi-${icon} ${color}"></i>
            <span class="text-truncate">${filename}</span>
            ${error ? `<span class="text-danger">(${error})</span>` : ''}
        </div>`;
    
    statusDiv.innerHTML += html;
}

function showStatus(type, message) {
    const status = document.getElementById('uploadStatus');
    const alertClass = type === 'error' ? 'danger' :
                      type === 'warning' ? 'warning' : 
                      type === 'info' ? 'info' : 'success';
    const icon = type === 'error' ? 'exclamation-triangle' :
                type === 'warning' ? 'exclamation-circle' :
                type === 'info' ? 'info-circle' : 'check-circle';

    status.innerHTML = `
        <div class="alert alert-${alertClass} p-2 mb-0 small fade-in">
            <i class="bi bi-${icon} me-2"></i>
            ${message.split('\n').join('<br>')}
        </div>`;
}

function updateDocumentsList(documents) {
    const documentsSelect = document.getElementById('documents');
    documentsSelect.innerHTML = documents
        .map(doc => `<option value="${doc}">${doc}</option>`)
        .join('');
}

function formatAnswer(text) {
    return text.split('\n')
        .map(line => line.trim())
        .filter(line => line)
        .map(line => `<p class="mb-2">${line}</p>`)
        .join('');
}

function formatError(message) {
    return `
        <div class="text-danger">
            <i class="bi bi-exclamation-triangle me-2"></i>${message}
        </div>`;
}