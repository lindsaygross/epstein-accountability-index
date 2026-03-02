// Attribution: Scaffolded with AI assistance (Claude, Anthropic)

/**
 * Main application JavaScript for The Accountability Gap dashboard.
 */

// Global state
let chartData = [];
let modelResults = {};
let experimentResults = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing dashboard...');

    // Load initial data
    loadChartData();
    loadModelResults();
    loadExperimentResults();

    // Setup event listeners
    setupEventListeners();
});

/**
 * Setup event listeners for interactive elements.
 */
function setupEventListeners() {
    const searchInput = document.getElementById('personSearch');
    const searchBtn = document.getElementById('searchBtn');

    // Search button click
    searchBtn.addEventListener('click', function() {
        const query = searchInput.value.trim();
        if (query) {
            searchPerson(query);
        }
    });

    // Search on Enter key
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                searchPerson(query);
            }
        }
    });

    // Live search suggestions
    searchInput.addEventListener('input', function() {
        const query = searchInput.value.trim();
        if (query.length >= 2) {
            fetchSearchSuggestions(query);
        } else {
            document.getElementById('searchResults').innerHTML = '';
        }
    });
}

/**
 * Fetch search suggestions from API.
 */
function fetchSearchSuggestions(query) {
    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(names => {
            displaySearchSuggestions(names);
        })
        .catch(error => {
            console.error('Search error:', error);
        });
}

/**
 * Display search suggestions.
 */
function displaySearchSuggestions(names) {
    const resultsDiv = document.getElementById('searchResults');

    if (names.length === 0) {
        resultsDiv.innerHTML = '<div class="p-2 text-muted">No results found</div>';
        return;
    }

    resultsDiv.innerHTML = names.map(name =>
        `<div class="search-result-item" onclick="loadPerson('${name}')">${name}</div>`
    ).join('');
}

/**
 * Search for a person and load their profile.
 */
function searchPerson(query) {
    // For simplicity, we'll search and load the first result
    fetch(`/api/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(names => {
            if (names.length > 0) {
                loadPerson(names[0]);
            } else {
                alert('No person found with that name.');
            }
        })
        .catch(error => {
            console.error('Search error:', error);
            alert('Error searching for person.');
        });
}

/**
 * Load and display person profile.
 */
function loadPerson(name) {
    fetch(`/api/person/${encodeURIComponent(name)}`)
        .then(response => response.json())
        .then(data => {
            displayPersonProfile(data);
            // Clear search
            document.getElementById('personSearch').value = '';
            document.getElementById('searchResults').innerHTML = '';
        })
        .catch(error => {
            console.error('Error loading person:', error);
            alert('Error loading person profile.');
        });
}

/**
 * Display person profile data.
 */
function displayPersonProfile(data) {
    // Show profile card
    document.getElementById('personProfile').style.display = 'block';

    // Set name
    document.getElementById('personName').textContent = data.name;

    // Create severity gauge
    createSeverityGauge(data.severity_score);
    document.getElementById('severityValue').textContent = data.severity_score.toFixed(2);

    // Set consequence badge
    const badge = document.getElementById('consequenceBadge');
    badge.className = `badge bg-${data.badge.color}`;
    badge.textContent = data.badge.label;

    // Set consequence description
    document.getElementById('consequenceDescription').textContent = data.consequence_description;

    // Display model predictions
    displayModelPredictions(data.predictions, data.consequence_tier);

    // Display features
    displayFeatures(data.features);
}

/**
 * Create severity score gauge chart.
 */
function createSeverityGauge(score) {
    const data = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: score,
        gauge: {
            axis: { range: [0, 10], tickcolor: '#e0e0e0' },
            bar: { color: '#0d6efd' },
            bgcolor: '#3d3d3d',
            borderwidth: 2,
            bordercolor: '#505050',
            steps: [
                { range: [0, 3], color: '#28a745' },
                { range: [3, 7], color: '#ffc107' },
                { range: [7, 10], color: '#dc3545' }
            ],
            threshold: {
                line: { color: 'white', width: 4 },
                thickness: 0.75,
                value: score
            }
        }
    }];

    const layout = {
        height: 200,
        margin: { t: 0, b: 0, l: 20, r: 20 },
        paper_bgcolor: '#2d2d2d',
        plot_bgcolor: '#2d2d2d',
        font: { color: '#e0e0e0' }
    };

    Plotly.newPlot('severityGauge', data, layout, {displayModeBar: false});
}

/**
 * Display model predictions.
 */
function displayModelPredictions(predictions, actualTier) {
    const container = document.getElementById('modelPredictions');

    const tierLabels = ['No Consequence', 'Soft', 'Hard'];

    let html = '<table class="table table-sm table-dark">';
    html += '<thead><tr><th>Model</th><th>Predicted</th><th>Actual</th></tr></thead>';
    html += '<tbody>';

    for (const [model, pred] of Object.entries(predictions)) {
        const predLabel = pred !== null ? tierLabels[pred] : 'N/A';
        const actualLabel = tierLabels[actualTier];
        const match = pred === actualTier;

        html += `<tr>
            <td>${model}</td>
            <td class="${match ? 'text-success' : 'text-danger'}">${predLabel}</td>
            <td>${actualLabel}</td>
        </tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

/**
 * Display feature values.
 */
function displayFeatures(features) {
    const container = document.getElementById('features');

    const featureLabels = {
        'mention_count': 'Documents',
        'total_mentions': 'Total Mentions',
        'mean_sentiment': 'Avg Sentiment',
        'cooccurrence_score': 'Co-occurrence',
        'doc_type_diversity': 'Doc Types',
        'in_subject_line': 'In Subject'
    };

    let html = '';
    for (const [key, value] of Object.entries(features)) {
        const label = featureLabels[key] || key;
        const displayValue = typeof value === 'boolean'
            ? (value ? 'Yes' : 'No')
            : typeof value === 'number'
            ? value.toFixed(2)
            : value;

        html += `
            <div class="col-md-4 mb-2">
                <div class="feature-item">
                    <div class="feature-label">${label}</div>
                    <div class="feature-value">${displayValue}</div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Load accountability gap chart data.
 */
function loadChartData() {
    fetch('/api/chart-data')
        .then(response => response.json())
        .then(data => {
            chartData = data;
            createScatterPlot(data);
        })
        .catch(error => {
            console.error('Error loading chart data:', error);
        });
}

/**
 * Create accountability gap scatter plot.
 */
function createScatterPlot(data) {
    // Group by power tier
    const tiers = ['Low', 'Medium', 'High', 'Very High'];
    const colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545'];

    const traces = tiers.map((tier, idx) => {
        const tierData = data.filter(d => d.power_tier === tier);

        return {
            x: tierData.map(d => d.severity_score),
            y: tierData.map(d => d.consequence_tier + (Math.random() - 0.5) * 0.2), // Add jitter
            mode: 'markers',
            type: 'scatter',
            name: tier,
            marker: {
                size: 10,
                color: colors[idx],
                opacity: 0.7,
                line: {
                    color: '#ffffff',
                    width: 1
                }
            },
            text: tierData.map(d => d.name),
            hovertemplate: '<b>%{text}</b><br>' +
                          'Severity: %{x:.2f}<br>' +
                          'Consequence: %{y:.0f}<br>' +
                          '<extra></extra>'
        };
    });

    const layout = {
        title: {
            text: 'Does Severity Predict Consequences?',
            font: { color: '#e0e0e0' }
        },
        xaxis: {
            title: 'Severity Score',
            color: '#e0e0e0',
            gridcolor: '#404040'
        },
        yaxis: {
            title: 'Consequence Tier',
            color: '#e0e0e0',
            gridcolor: '#404040',
            tickvals: [0, 1, 2],
            ticktext: ['None', 'Soft', 'Hard']
        },
        paper_bgcolor: '#2d2d2d',
        plot_bgcolor: '#2d2d2d',
        font: { color: '#e0e0e0' },
        hovermode: 'closest',
        height: 500,
        margin: { t: 50, b: 50, l: 60, r: 20 },
        legend: {
            bgcolor: '#3d3d3d',
            bordercolor: '#505050',
            borderwidth: 1
        }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot('scatterPlot', traces, layout, config);
}

/**
 * Load model performance results.
 */
function loadModelResults() {
    fetch('/api/model-results')
        .then(response => response.json())
        .then(data => {
            modelResults = data;
            displayModelResults(data);
        })
        .catch(error => {
            console.error('Error loading model results:', error);
        });
}

/**
 * Display model performance metrics.
 */
function displayModelResults(data) {
    const container = document.getElementById('modelPerformance');

    const modelLabels = {
        'naive_baseline': 'Naive Baseline',
        'xgboost': 'XGBoost',
        'distilbert': 'DistilBERT'
    };

    let html = '<div class="performance-table">';

    for (const [model, metrics] of Object.entries(data)) {
        const label = modelLabels[model] || model;
        html += `
            <div class="performance-row">
                <div>
                    <div class="model-name">${label}</div>
                    <small class="text-muted">F1: ${(metrics.f1_macro * 100).toFixed(1)}%</small>
                </div>
                <div class="model-metric">${(metrics.accuracy * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    html += '</div>';
    container.innerHTML = html;
}

/**
 * Load experiment results.
 */
function loadExperimentResults() {
    fetch('/api/experiment-results')
        .then(response => response.json())
        .then(data => {
            experimentResults = data;
            createExperimentChart(data);
        })
        .catch(error => {
            console.error('Error loading experiment results:', error);
        });
}

/**
 * Create experiment results bar chart.
 */
function createExperimentChart(data) {
    const trace = {
        x: data.map(d => d.power_tier),
        y: data.map(d => d.correlation),
        type: 'bar',
        marker: {
            color: data.map(d => d.correlation > 0.5 ? '#28a745' : d.correlation > 0.3 ? '#ffc107' : '#dc3545'),
            line: {
                color: '#ffffff',
                width: 1
            }
        },
        text: data.map(d => d.correlation.toFixed(3)),
        textposition: 'outside',
        hovertemplate: '<b>%{x}</b><br>' +
                      'Correlation: %{y:.3f}<br>' +
                      'N: %{customdata}<br>' +
                      '<extra></extra>',
        customdata: data.map(d => d.n_people)
    };

    const layout = {
        yaxis: {
            title: 'Correlation',
            color: '#e0e0e0',
            gridcolor: '#404040',
            range: [-0.2, 1.0]
        },
        xaxis: {
            color: '#e0e0e0'
        },
        paper_bgcolor: '#2d2d2d',
        plot_bgcolor: '#2d2d2d',
        font: { color: '#e0e0e0', size: 10 },
        height: 300,
        margin: { t: 20, b: 40, l: 50, r: 20 },
        showlegend: false
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot('experimentChart', [trace], layout, config);
}
