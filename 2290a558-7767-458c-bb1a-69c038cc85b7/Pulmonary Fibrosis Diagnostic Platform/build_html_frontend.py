import os

# Create comprehensive HTML templates for the frontend

# 1. Main Dashboard (index.html)
index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaggle Dataset Manager</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Kaggle Dataset Manager</h1>
            <p class="subtitle">Download, preview, and analyze Kaggle datasets</p>
        </header>

        <nav class="tabs">
            <button class="tab-btn active" data-tab="download">Download Dataset</button>
            <button class="tab-btn" data-tab="datasets">My Datasets</button>
            <button class="tab-btn" data-tab="preview">Preview Data</button>
            <button class="tab-btn" data-tab="statistics">Statistics</button>
        </nav>

        <!-- Download Dataset Tab -->
        <div class="tab-content active" id="download-tab">
            <div class="card">
                <h2>Download Kaggle Dataset</h2>
                <form id="download-form">
                    <div class="form-group">
                        <label for="dataset-id">Dataset ID</label>
                        <input type="text" id="dataset-id" name="dataset_id" 
                               placeholder="e.g., datasnaek/youtube-new" required>
                        <small>Format: username/dataset-name (found in Kaggle dataset URL)</small>
                    </div>
                    <button type="submit" class="btn btn-primary">Download Dataset</button>
                </form>
                <div id="download-result" class="result-box"></div>
            </div>
        </div>

        <!-- My Datasets Tab -->
        <div class="tab-content" id="datasets-tab">
            <div class="card">
                <h2>Downloaded Datasets</h2>
                <button class="btn btn-secondary" onclick="loadDatasets()">Refresh List</button>
                <div id="datasets-list" class="datasets-list"></div>
            </div>
        </div>

        <!-- Preview Data Tab -->
        <div class="tab-content" id="preview-tab">
            <div class="card">
                <h2>Preview Data File</h2>
                <form id="preview-form">
                    <div class="form-group">
                        <label for="preview-file">File Path</label>
                        <input type="text" id="preview-file" name="file_path" 
                               placeholder="e.g., datasnaek_youtube-new/USvideos.csv" required>
                        <small>Relative path from data directory</small>
                    </div>
                    <div class="form-group">
                        <label for="num-rows">Number of Rows</label>
                        <input type="number" id="num-rows" name="num_rows" 
                               value="10" min="1" max="100">
                    </div>
                    <button type="submit" class="btn btn-primary">Preview File</button>
                    <button type="button" class="btn btn-secondary" onclick="loadFiles()">Browse Files</button>
                </form>
                <div id="preview-result" class="result-box"></div>
            </div>
        </div>

        <!-- Statistics Tab -->
        <div class="tab-content" id="statistics-tab">
            <div class="card">
                <h2>Dataset Statistics</h2>
                <form id="statistics-form">
                    <div class="form-group">
                        <label for="stats-file">File Path</label>
                        <input type="text" id="stats-file" name="file_path" 
                               placeholder="e.g., datasnaek_youtube-new/USvideos.csv" required>
                        <small>Relative path from data directory</small>
                    </div>
                    <button type="submit" class="btn btn-primary">Get Statistics</button>
                    <button type="button" class="btn btn-secondary" onclick="loadFiles()">Browse Files</button>
                </form>
                <div id="statistics-result" class="result-box"></div>
            </div>
        </div>

        <footer>
            <p>Powered by FastAPI & Kaggle API | <a href="/docs" target="_blank">API Documentation</a></p>
        </footer>
    </div>

    <script src="/static/js/main.js"></script>
</body>
</html>
'''

with open('templates/index.html', 'w') as f:
    f.write(index_html)
print("✓ Created templates/index.html")

# 2. Enhanced CSS Styling
css_content = '''/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* Zerve Design System Colors */
    --bg-primary: #1D1D20;
    --bg-secondary: #2a2a2e;
    --bg-card: #35353a;
    --text-primary: #fbfbff;
    --text-secondary: #909094;
    --accent-blue: #A1C9F4;
    --accent-orange: #FFB482;
    --accent-green: #8DE5A1;
    --accent-coral: #FF9F9B;
    --accent-highlight: #ffd400;
    --success: #17b26a;
    --error: #f04438;
    --border: #404046;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
header {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
    border-radius: 12px;
    margin-bottom: 30px;
    border: 1px solid var(--border);
}

header h1 {
    font-size: 2.5rem;
    color: var(--accent-blue);
    margin-bottom: 10px;
}

.subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
}

/* Tabs Navigation */
.tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 30px;
    border-bottom: 2px solid var(--border);
    flex-wrap: wrap;
}

.tab-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    padding: 15px 25px;
    font-size: 1rem;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.3s ease;
}

.tab-btn:hover {
    color: var(--text-primary);
    background: var(--bg-secondary);
}

.tab-btn.active {
    color: var(--accent-blue);
    border-bottom-color: var(--accent-blue);
}

/* Tab Content */
.tab-content {
    display: none;
    animation: fadeIn 0.3s ease;
}

.tab-content.active {
    display: block;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.card h2 {
    color: var(--accent-blue);
    margin-bottom: 20px;
    font-size: 1.8rem;
}

/* Forms */
.form-group {
    margin-bottom: 20px;
}

.form-group label {
    display: block;
    color: var(--text-primary);
    margin-bottom: 8px;
    font-weight: 500;
}

.form-group input,
.form-group select {
    width: 100%;
    padding: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group select:focus {
    outline: none;
    border-color: var(--accent-blue);
}

.form-group small {
    display: block;
    color: var(--text-secondary);
    margin-top: 5px;
    font-size: 0.85rem;
}

/* Buttons */
.btn {
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
    margin-right: 10px;
    margin-bottom: 10px;
}

.btn-primary {
    background: var(--accent-blue);
    color: var(--bg-primary);
}

.btn-primary:hover {
    background: #8ab5e0;
    transform: translateY(-2px);
}

.btn-secondary {
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
}

.btn-secondary:hover {
    background: var(--bg-primary);
}

/* Result Boxes */
.result-box {
    margin-top: 20px;
    padding: 20px;
    border-radius: 8px;
    display: none;
}

.result-box.success {
    display: block;
    background: rgba(23, 178, 106, 0.1);
    border: 1px solid var(--success);
}

.result-box.error {
    display: block;
    background: rgba(240, 68, 56, 0.1);
    border: 1px solid var(--error);
    color: var(--error);
}

.result-box.info {
    display: block;
    background: rgba(161, 201, 244, 0.1);
    border: 1px solid var(--accent-blue);
}

/* Tables */
.data-table {
    width: 100%;
    margin-top: 15px;
    border-collapse: collapse;
    overflow-x: auto;
    display: block;
}

.data-table th,
.data-table td {
    padding: 12px;
    text-align: left;
    border: 1px solid var(--border);
}

.data-table th {
    background: var(--bg-secondary);
    color: var(--accent-blue);
    font-weight: 600;
}

.data-table tr:hover {
    background: var(--bg-secondary);
}

/* Dataset List */
.datasets-list {
    margin-top: 20px;
}

.dataset-item {
    background: var(--bg-secondary);
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 8px;
    border-left: 4px solid var(--accent-green);
}

.dataset-item h3 {
    color: var(--accent-green);
    margin-bottom: 8px;
}

.dataset-item p {
    color: var(--text-secondary);
    margin: 5px 0;
}

/* Loading Spinner */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Footer */
footer {
    text-align: center;
    padding: 30px 20px;
    color: var(--text-secondary);
    margin-top: 50px;
    border-top: 1px solid var(--border);
}

footer a {
    color: var(--accent-blue);
    text-decoration: none;
}

footer a:hover {
    text-decoration: underline;
}

/* Responsive Design */
@media (max-width: 768px) {
    header h1 {
        font-size: 2rem;
    }
    
    .tabs {
        flex-direction: column;
    }
    
    .tab-btn {
        width: 100%;
    }
    
    .card {
        padding: 20px;
    }
}

/* Utility Classes */
.text-success { color: var(--success); }
.text-error { color: var(--error); }
.text-warning { color: var(--accent-orange); }
.text-info { color: var(--accent-blue); }
.mt-20 { margin-top: 20px; }
.mb-20 { margin-bottom: 20px; }
'''

with open('static/css/style.css', 'w') as f:
    f.write(css_content)
print("✓ Created static/css/style.css")

# 3. JavaScript for frontend interactivity
js_content = '''// API Base URL
const API_BASE = '/api/v1';

// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            
            // Remove active class from all tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            btn.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
    
    // Form submissions
    setupDownloadForm();
    setupPreviewForm();
    setupStatisticsForm();
});

// Download Dataset Form
function setupDownloadForm() {
    const form = document.getElementById('download-form');
    const resultDiv = document.getElementById('download-result');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const datasetId = document.getElementById('dataset-id').value;
        resultDiv.className = 'result-box info';
        resultDiv.innerHTML = '<span class="loading"></span> Downloading dataset...';
        
        try {
            const response = await fetch(`${API_BASE}/download`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: datasetId })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                resultDiv.className = 'result-box success';
                resultDiv.innerHTML = `
                    <h3>✅ Download Successful!</h3>
                    <p><strong>Dataset:</strong> ${data.dataset_id}</p>
                    <p><strong>Location:</strong> ${data.download_path}</p>
                    <p><strong>Files:</strong></p>
                    <ul>${data.files.map(f => `<li>${f}</li>`).join('')}</ul>
                `;
            } else {
                throw new Error(data.detail || 'Download failed');
            }
        } catch (error) {
            resultDiv.className = 'result-box error';
            resultDiv.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
        }
    });
}

// Preview File Form
function setupPreviewForm() {
    const form = document.getElementById('preview-form');
    const resultDiv = document.getElementById('preview-result');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const filePath = document.getElementById('preview-file').value;
        const numRows = document.getElementById('num-rows').value;
        
        resultDiv.className = 'result-box info';
        resultDiv.innerHTML = '<span class="loading"></span> Loading preview...';
        
        try {
            const response = await fetch(`${API_BASE}/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath, num_rows: parseInt(numRows) })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                resultDiv.className = 'result-box success';
                resultDiv.innerHTML = `
                    <h3>📄 File Preview: ${data.file_name}</h3>
                    <p><strong>Rows:</strong> ${data.num_rows} | <strong>Columns:</strong> ${data.num_columns}</p>
                    ${formatPreviewTable(data)}
                `;
            } else {
                throw new Error(data.detail || 'Preview failed');
            }
        } catch (error) {
            resultDiv.className = 'result-box error';
            resultDiv.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
        }
    });
}

// Statistics Form
function setupStatisticsForm() {
    const form = document.getElementById('statistics-form');
    const resultDiv = document.getElementById('statistics-result');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const filePath = document.getElementById('stats-file').value;
        
        resultDiv.className = 'result-box info';
        resultDiv.innerHTML = '<span class="loading"></span> Calculating statistics...';
        
        try {
            const response = await fetch(`${API_BASE}/statistics`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                resultDiv.className = 'result-box success';
                resultDiv.innerHTML = formatStatistics(data);
            } else {
                throw new Error(data.detail || 'Statistics calculation failed');
            }
        } catch (error) {
            resultDiv.className = 'result-box error';
            resultDiv.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
        }
    });
}

// Format preview table
function formatPreviewTable(data) {
    let html = '<h4>Column Info:</h4><table class="data-table"><thead><tr>';
    html += '<th>Column</th><th>Type</th></tr></thead><tbody>';
    
    for (const [col, type] of Object.entries(data.dtypes)) {
        html += `<tr><td>${col}</td><td>${type}</td></tr>`;
    }
    html += '</tbody></table>';
    
    html += '<h4 class="mt-20">Preview Data:</h4><table class="data-table"><thead><tr>';
    data.columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    data.preview.forEach(row => {
        html += '<tr>';
        data.columns.forEach(col => {
            html += `<td>${row[col] !== null ? row[col] : 'null'}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    
    return html;
}

// Format statistics
function formatStatistics(data) {
    let html = `<h3>📊 Dataset Statistics: ${data.file_name}</h3>`;
    html += `<p><strong>Rows:</strong> ${data.basic_info.row_count.toLocaleString()} | `;
    html += `<strong>Columns:</strong> ${data.basic_info.column_count} | `;
    html += `<strong>Size:</strong> ${data.basic_info.file_size} | `;
    html += `<strong>Memory:</strong> ${data.basic_info.memory_usage}</p>`;
    
    // Missing values
    html += '<h4 class="mt-20">Missing Values:</h4><table class="data-table"><thead><tr>';
    html += '<th>Column</th><th>Missing Count</th><th>Percentage</th></tr></thead><tbody>';
    for (const [col, count] of Object.entries(data.missing_values.counts)) {
        const pct = data.missing_values.percentages[col];
        html += `<tr><td>${col}</td><td>${count}</td><td>${pct}%</td></tr>`;
    }
    html += '</tbody></table>';
    
    // Numeric statistics
    if (Object.keys(data.numeric_stats).length > 0) {
        html += '<h4 class="mt-20">Numeric Columns:</h4>';
        for (const [col, stats] of Object.entries(data.numeric_stats)) {
            html += `<div class="dataset-item"><h3>${col}</h3>`;
            html += `<p>Mean: ${stats.mean} | Median: ${stats.median} | Std: ${stats.std}</p>`;
            html += `<p>Min: ${stats.min} | Max: ${stats.max}</p></div>`;
        }
    }
    
    return html;
}

// Load datasets list
async function loadDatasets() {
    const listDiv = document.getElementById('datasets-list');
    listDiv.innerHTML = '<span class="loading"></span> Loading datasets...';
    
    try {
        const response = await fetch(`${API_BASE}/datasets`);
        const data = await response.json();
        
        if (data.datasets.length === 0) {
            listDiv.innerHTML = '<p class="text-secondary">No datasets downloaded yet.</p>';
            return;
        }
        
        listDiv.innerHTML = '';
        data.datasets.forEach(dataset => {
            const div = document.createElement('div');
            div.className = 'dataset-item';
            div.innerHTML = `
                <h3>${dataset.name}</h3>
                <p><strong>Path:</strong> ${dataset.path}</p>
                <p><strong>Files:</strong> ${dataset.file_count} | <strong>Size:</strong> ${dataset.total_size}</p>
            `;
            listDiv.appendChild(div);
        });
    } catch (error) {
        listDiv.innerHTML = `<p class="text-error">Error loading datasets: ${error.message}</p>`;
    }
}

// Load files list
async function loadFiles() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();
        
        if (data.files.length === 0) {
            alert('No data files found. Download a dataset first.');
            return;
        }
        
        let fileList = 'Available files:\\n\\n';
        data.files.forEach((file, idx) => {
            fileList += `${idx + 1}. ${file.path} (${file.size})\\n`;
        });
        
        alert(fileList);
    } catch (error) {
        alert(`Error loading files: ${error.message}`);
    }
}
'''

with open('static/js/main.js', 'w') as f:
    f.write(js_content)
print("✓ Created static/js/main.js")

print("\n✅ HTML/CSS Frontend Implementation Complete!")
print("\nCreated Files:")
print("  ✓ templates/index.html - Main dashboard with tabs")
print("  ✓ static/css/style.css - Zerve design system styling")
print("  ✓ static/js/main.js - Interactive frontend functionality")
print("\nFeatures:")
print("  ✓ Dataset download interface")
print("  ✓ File preview with data type display")
print("  ✓ Statistics dashboard with comprehensive metrics")
print("  ✓ Dataset browser and file explorer")
print("  ✓ Responsive design with Zerve color palette")
print("  ✓ Tab-based navigation for clean UX")
print("  ✓ Real-time API communication")
print("  ✓ Professional styling suitable for reports")

frontend_status = "complete"
