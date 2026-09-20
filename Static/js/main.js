// PlaylistPulse UI Interactions

document.addEventListener('DOMContentLoaded', function() {
    
    // 1. Setup Live Slider Label Listeners
    setupSliders();

    // 2. Setup Prediction Form AJAX Submission
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            runPrediction();
        });
    }
});

/**
 * Syncs range inputs (sliders) with their corresponding text labels.
 */
function setupSliders() {
    const sliders = [
        { id: 'danceability', label: 'dance_val', suffix: '' },
        { id: 'energy', label: 'energy_val', suffix: '' },
        { id: 'acousticness', label: 'acoustic_val', suffix: '' },
        { id: 'speechiness', label: 'speech_val', suffix: '' },
        { id: 'instrumentalness', label: 'instrumental_val', suffix: '' },
        { id: 'liveness', label: 'liveness_val', suffix: '' },
        { id: 'valence', label: 'valence_val', suffix: '' },
        { id: 'loudness', label: 'loudness_val', suffix: ' dB' }
    ];

    sliders.forEach(slider => {
        const inputEl = document.getElementById(slider.id);
        const labelEl = document.getElementById(slider.label);
        
        if (inputEl && labelEl) {
            inputEl.addEventListener('input', function() {
                labelEl.textContent = this.value + slider.suffix;
            });
        }
    });
}

/**
 * Submits form data via AJAX, animates the prediction trace, and renders results.
 */
async function runPrediction() {
    const form = document.getElementById('predictionForm');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const resultsContent = document.getElementById('resultsContent');
    
    if (!form) return;
    
    // Retrieve form values
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        // Convert numbers and boolean checkbox
        if (key === 'explicit') {
            data[key] = true;
        } else if (['duration_ms', 'danceability', 'energy', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo', 'key', 'mode', 'time_signature'].includes(key)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    });
    
    // Explicit handle checkbox false state (if not checked it won't be in formData)
    if (!data.hasOwnProperty('explicit')) {
        data['explicit'] = false;
    }
    
    // Convert duration seconds to milliseconds for the API
    data['duration_ms'] = data['duration_ms'] * 1000;
    
    // Get chosen models
    const regModel = document.getElementById('reg_model').value;
    const clfModel = document.getElementById('clf_model').value;
    
    const requestPayload = {
        features: data,
        regression_model: regModel,
        classification_model: clfModel
    };

    // Reset flowchart classes
    resetTraceFlow();
    
    // Hide results during prediction
    resultsContent.classList.add('d-none');
    resultsPlaceholder.classList.remove('d-none');
    resultsPlaceholder.innerHTML = `
        <div class="spinner-border text-success mb-3" role="status"></div>
        <p class="mb-0 text-success fw-semibold">Processing Prediction Trace...</p>
    `;

    try {
        // Trace Step 1: Input (UI side)
        await activateTraceNode('node-input', null);
        await delay(200);

        // Trace Step 2: API Route call
        await activateTraceNode('node-api', 'node-input');
        
        // Dispatch Fetch request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestPayload)
        });
        
        if (!response.ok) throw new Error('API server returned an error');
        const result = await response.json();
        
        await delay(200);
        
        // Trace Step 3: Schema Validation
        await activateTraceNode('node-validation', 'node-api');
        await delay(200);
        
        // Trace Step 4: Preprocessing
        await activateTraceNode('node-preprocessing', 'node-validation');
        await delay(200);
        
        // Trace Step 5: Model prediction
        await activateTraceNode('node-model', 'node-preprocessing');
        await delay(200);
        
        // Trace Step 6: Database Logging
        await activateTraceNode('node-logging', 'node-model');
        await delay(200);
        
        // Trace Step 7: Response Render
        await activateTraceNode('node-response', 'node-logging');
        await delay(200);
        
        // Render values to UI elements
        document.getElementById('pred_popularity').textContent = Math.round(result.popularity_prediction);
        
        const skipProb = result.skip_prediction_probability;
        document.getElementById('pred_skip_prob').textContent = (skipProb * 100).toFixed(1) + '%';
        document.getElementById('skip_pct_label').textContent = (skipProb * 100).toFixed(0) + '%';
        
        const progressBar = document.getElementById('skip_progress_bar');
        progressBar.style.width = (skipProb * 100) + '%';
        
        // Color coding for progress bar & risk category badge
        const riskCategory = result.skip_risk_category;
        const badge = document.getElementById('skip_risk_badge');
        badge.textContent = riskCategory;
        
        // Reset classes
        badge.className = "badge fs-5 py-2 px-3 mb-2";
        progressBar.className = "progress-bar progress-bar-striped progress-bar-animated";
        
        if (riskCategory === 'High') {
            badge.classList.add('bg-danger');
            progressBar.classList.add('bg-danger');
        } else if (riskCategory === 'Medium') {
            badge.classList.add('bg-warning', 'text-dark');
            progressBar.classList.add('bg-warning');
        } else {
            badge.classList.add('bg-success');
            progressBar.classList.add('bg-success');
        }
        
        // Set metadata info
        document.getElementById('meta_reg_model').textContent = result.models_used.regression;
        document.getElementById('meta_clf_model').textContent = result.models_used.classification;
        document.getElementById('meta_version').textContent = result.model_version;
        
        // Show results container
        resultsPlaceholder.classList.add('d-none');
        resultsContent.classList.remove('d-none');
        
    } catch (error) {
        console.error(error);
        resultsPlaceholder.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation display-3 mb-3 text-danger"></i>
            <p class="mb-0 text-danger fw-semibold">Prediction Failed: ${error.message}</p>
        `;
    }
}

/**
 * Highlights a flowchart node and connects the arrow path from previous.
 */
async function activateTraceNode(nodeId, prevNodeId) {
    // If there's a previous node, light up the connecting arrow
    if (prevNodeId) {
        const prevNode = document.getElementById(prevNodeId);
        if (prevNode) {
            const nextArrow = prevNode.nextElementSibling;
            if (nextArrow && nextArrow.classList.contains('trace-arrow')) {
                nextArrow.classList.add('active-arrow');
            }
        }
    }
    
    // Light up current node
    const node = document.getElementById(nodeId);
    if (node) {
        node.classList.add('active-node');
    }
}

/**
 * Resets trace classes to initial gray state.
 */
function resetTraceFlow() {
    const nodes = document.querySelectorAll('.trace-node');
    nodes.forEach(node => {
        node.classList.remove('active-node');
    });
    
    const arrows = document.querySelectorAll('.trace-arrow');
    arrows.forEach(arrow => {
        arrow.classList.remove('active-arrow');
    });
}

/**
 * Utility helper to pause execution.
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
