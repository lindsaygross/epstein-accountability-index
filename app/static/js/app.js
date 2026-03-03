// Attribution: Scaffolded with AI assistance (Claude, Anthropic)

/**
 * The Accountability Gap — Main Application JavaScript
 *
 * Sections:
 *   1. Global State & Constants
 *   2. Data Loading (parallel fetch)
 *   3. D3.js Network Graph
 *   4. People Grid (search, filter, sort)
 *   5. Person Modal
 *   6. Analysis Charts (Plotly, restyled)
 *   7. Side Navigation (IntersectionObserver)
 *   8. GSAP Scroll Animations
 *   9. Initialization
 */

/* ============================================================
   1. GLOBAL STATE & CONSTANTS
   ============================================================ */

const STATE = {
    people: [],
    edges: [],
    chartData: [],
    modelResults: {},
    experimentResults: [],
    ablationResults: [],
    activeFilter: 'all',
    activeSort: 'severity',
    searchQuery: ''
};

const SEVERITY_COLORS = {
    Critical: '#ff3333',
    High:     '#ff6b35',
    Medium:   '#ffc107',
    Low:      '#4ecdc4',
    Minimal:  '#555555'
};

const PLOTLY_DARK = {
    paper_bgcolor: '#111111',
    plot_bgcolor:  '#111111',
    font: { color: '#aaa', family: 'Inter, sans-serif', size: 11 }
};

/* ============================================================
   2. DATA LOADING
   ============================================================ */

async function loadAllData() {
    try {
        const [people, edges, chartData, modelResults, experimentResults, ablationResults] =
            await Promise.all([
                fetch('/api/people').then(r => r.json()),
                fetch('/api/edges').then(r => r.json()),
                fetch('/api/chart-data').then(r => r.json()),
                fetch('/api/model-results').then(r => r.json()).catch(() => ({})),
                fetch('/api/experiment-results').then(r => r.json()).catch(() => []),
                fetch('/api/ablation-results').then(r => r.json()).catch(() => [])
            ]);

        STATE.people = people;
        STATE.edges = edges;
        STATE.chartData = chartData;
        STATE.modelResults = modelResults;
        STATE.experimentResults = Array.isArray(experimentResults) ? experimentResults : [];
        STATE.ablationResults = Array.isArray(ablationResults) ? ablationResults : [];

        console.log(`Loaded ${people.length} people, ${edges.length} edges`);

        // Render everything
        initNetworkGraph();
        renderPeopleGrid();
        createScatterPlot();
        displayModelResults();
        createExperimentChart();
        createAblationChart();

    } catch (err) {
        console.error('Failed to load data:', err);
    }
}

/* ============================================================
   3. D3.js NETWORK GRAPH
   ============================================================ */

function initNetworkGraph() {
    const container = document.getElementById('network-graph');
    if (!container || !STATE.people.length) return;

    const width  = container.clientWidth  || 900;
    const height = container.clientHeight || Math.max(600, window.innerHeight * 0.7);

    // Clear previous
    container.innerHTML = '';

    // Build node and link data
    const nameSet = new Set(STATE.people.map(p => p.name));

    const nodes = STATE.people.map(p => ({
        id:          p.name,
        severity:    p.severity_score,
        level:       p.level,
        consequence: p.consequence_tier,
        radius:      Math.max(12, Math.sqrt(p.severity_score) * 6),
        image_url:   p.image_url || ''
    }));

    const links = STATE.edges
        .filter(e => nameSet.has(e.source) && nameSet.has(e.target))
        .map(e => ({ source: e.source, target: e.target, weight: e.weight }));

    const maxWeight = d3.max(links, d => d.weight) || 1;

    // SVG with zoom
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g');

    svg.call(
        d3.zoom()
            .scaleExtent([0.25, 5])
            .on('zoom', (event) => g.attr('transform', event.transform))
    );

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(90).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-180))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.radius + 3));

    // Links
    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', '#333')
        .attr('stroke-opacity', d => 0.12 + (d.weight / maxWeight) * 0.5)
        .attr('stroke-width', d => 0.4 + (d.weight / maxWeight) * 3);

    // Defs: circular clip paths for images
    const defs = svg.append('defs');
    nodes.forEach((d, i) => {
        defs.append('clipPath')
            .attr('id', `node-clip-${i}`)
            .append('circle')
            .attr('cx', 0).attr('cy', 0)
            .attr('r', d.radius - 1.5);
    });

    // Node groups (g) with image + ring
    const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('cursor', 'pointer')
        .call(makeDrag(simulation));

    // Severity-colored ring
    node.append('circle')
        .attr('r', d => d.radius)
        .attr('fill', '#111')
        .attr('stroke', d => SEVERITY_COLORS[d.level] || '#555')
        .attr('stroke-width', 2);

    // Person image inside clip
    node.append('image')
        .attr('href', d => d.image_url)
        .attr('x', d => -d.radius + 1.5)
        .attr('y', d => -d.radius + 1.5)
        .attr('width', d => (d.radius - 1.5) * 2)
        .attr('height', d => (d.radius - 1.5) * 2)
        .attr('clip-path', (d, i) => `url(#node-clip-${i})`)
        .attr('preserveAspectRatio', 'xMidYMid slice');

    // Labels — only show for higher-severity or larger nodes
    const label = g.append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(nodes)
        .join('text')
        .text(d => d.id)
        .attr('font-size', d => Math.max(8, d.radius * 0.75))
        .attr('fill', '#bbb')
        .attr('text-anchor', 'middle')
        .attr('dy', d => d.radius + 12)
        .attr('pointer-events', 'none')
        .style('font-family', 'Inter, sans-serif')
        .style('opacity', d => d.severity >= 5 ? 1 : 0);

    // Hover: highlight connections
    node.on('mouseover', function (event, d) {
        const connected = new Set([d.id]);
        links.forEach(l => {
            const src = typeof l.source === 'object' ? l.source.id : l.source;
            const tgt = typeof l.target === 'object' ? l.target.id : l.target;
            if (src === d.id) connected.add(tgt);
            if (tgt === d.id) connected.add(src);
        });

        node.attr('opacity', n => connected.has(n.__data__.id) ? 1 : 0.12);
        link.attr('opacity', l => {
            const src = typeof l.source === 'object' ? l.source.id : l.source;
            const tgt = typeof l.target === 'object' ? l.target.id : l.target;
            return (src === d.id || tgt === d.id) ? 0.85 : 0.02;
        });
        label.style('opacity', n => connected.has(n.id) ? 1 : 0);
        d3.select(this).select('circle').attr('stroke', '#fff').attr('stroke-width', 3);
    })
    .on('mouseout', function () {
        node.attr('opacity', 1);
        link.attr('stroke-opacity', d => 0.12 + (d.weight / maxWeight) * 0.5);
        link.attr('opacity', null);
        label.style('opacity', d => d.severity >= 5 ? 1 : 0);
        d3.select(this).select('circle').attr('stroke', d => SEVERITY_COLORS[d.level] || '#555').attr('stroke-width', 2);
    })
    .on('click', (event, d) => openPersonModal(d.id));

    // Simulation tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        node
            .attr('transform', d => `translate(${d.x},${d.y})`);
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
}

/** D3 drag behavior factory */
function makeDrag(simulation) {
    return d3.drag()
        .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        })
        .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
        })
        .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        });
}

/* ============================================================
   4. PEOPLE GRID
   ============================================================ */

function renderPeopleGrid() {
    const container = document.getElementById('people-grid');
    if (!container) return;

    let list = [...STATE.people];

    // Search filter
    if (STATE.searchQuery) {
        const q = STATE.searchQuery.toLowerCase();
        list = list.filter(p => p.name.toLowerCase().includes(q));
    }

    // Severity filter
    if (STATE.activeFilter !== 'all') {
        list = list.filter(p => p.level.toLowerCase() === STATE.activeFilter);
    }

    // Sort
    switch (STATE.activeSort) {
        case 'severity':
            list.sort((a, b) => b.severity_score - a.severity_score);
            break;
        case 'name':
            list.sort((a, b) => a.name.localeCompare(b.name));
            break;
        case 'consequence':
            list.sort((a, b) =>
                (b.consequence_tier - a.consequence_tier) ||
                (b.severity_score - a.severity_score)
            );
            break;
    }

    if (list.length === 0) {
        container.innerHTML =
            '<p style="text-align:center; color:#555; grid-column:1/-1; padding:3rem;">No individuals match your criteria.</p>';
        return;
    }

    container.innerHTML = list.map(p => {
        const lvl = p.level.toLowerCase();
        const color = SEVERITY_COLORS[p.level] || '#555';
        const cLabel = p.badge ? p.badge.label : '';
        const safeName = p.name.replace(/'/g, "\\'");

        const imgUrl = p.image_url || '/static/images/people/placeholder.png';
        return `
            <div class="person-card severity-${lvl}" role="listitem"
                 onclick="openPersonModal('${safeName}')"
                 tabindex="0"
                 aria-label="${p.name}, severity ${p.severity_score.toFixed(1)}">
                <img src="${imgUrl}" alt="" class="card-avatar"
                     onerror="this.src='/static/images/people/placeholder.png'" />
                <div class="name">${p.name}</div>
                <div class="score" style="color:${color}">${p.severity_score.toFixed(1)}</div>
                <span class="level-badge" style="color:${color}">${p.level}</span>
                <span class="consequence-badge">${cLabel}</span>
            </div>`;
    }).join('');
}

function setupPeopleControls() {
    // Live search
    const searchInput = document.getElementById('personSearch');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            STATE.searchQuery = searchInput.value.trim();
            renderPeopleGrid();
        });
    }

    // Filter pills
    document.querySelectorAll('.pill[data-filter]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            STATE.activeFilter = btn.dataset.filter;
            renderPeopleGrid();
        });
    });

    // Sort select
    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', () => {
            STATE.activeSort = sortSelect.value;
            renderPeopleGrid();
        });
    }
}

/* ============================================================
   5. PERSON MODAL
   ============================================================ */

function openPersonModal(name) {
    fetch(`/api/person/${encodeURIComponent(name)}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) { console.warn('Person not found:', name); return; }
            populateModal(data);
        })
        .catch(err => console.error('Error loading person:', err));
}

function populateModal(data) {
    const modal = document.getElementById('person-modal');
    if (!modal) return;

    // Header image
    const imgEl = document.getElementById('modal-image');
    if (imgEl) {
        imgEl.src = data.image_url || '/static/images/people/placeholder.png';
        imgEl.alt = data.name;
        imgEl.onerror = function() { this.src = '/static/images/people/placeholder.png'; };
    }

    // Header name & badge
    document.getElementById('modal-name').textContent = data.name;
    const badge = document.getElementById('modal-level');
    badge.textContent = data.level;
    badge.className = 'modal-severity-badge';
    badge.style.color = SEVERITY_COLORS[data.level] || '#555';

    // Gauge
    createModalGauge(data.severity_score);

    // Consequence
    const cEl = document.getElementById('modal-consequence');
    cEl.innerHTML = `
        <div style="margin-bottom:0.6rem;">
            <span class="consequence-badge" style="font-size:0.85rem; padding:0.3rem 0.8rem;
                color:${data.badge?.color === 'hard' ? '#ff3333' : data.badge?.color === 'soft' ? '#ffc107' : '#555'};">
                ${data.badge?.label || 'Unknown'}
            </span>
        </div>
        <p style="font-size:0.88rem; color:#999; line-height:1.65;">
            ${data.consequence_description || 'No information available'}
        </p>`;

    // NLP Features
    const fEl = document.getElementById('modal-features');
    const fLabels = {
        mention_count: 'Documents',
        total_mentions: 'Total Mentions',
        mean_sentiment: 'Avg Sentiment',
        cooccurrence_score: 'Co-occurrence',
        doc_type_diversity: 'Doc Types',
        in_subject_line: 'In Subject'
    };
    if (data.features) {
        fEl.innerHTML = Object.entries(data.features).map(([k, v]) => {
            let display = v;
            if (typeof v === 'boolean') display = v ? 'Yes' : 'No';
            else if (typeof v === 'number') display = Number.isInteger(v) ? v : v.toFixed(3);
            return `<div class="feature-item">
                <div class="feature-label">${fLabels[k] || k}</div>
                <div class="feature-value">${display}</div>
            </div>`;
        }).join('');
    }

    // Predictions
    const pEl = document.getElementById('modal-predictions');
    const tierText = { 0: 'No Consequence', 1: 'Consequence' };
    const mLabels = {
        logistic_baseline: 'Logistic Regression',
        random_forest_tfidf: 'Random Forest + TF-IDF',
        legal_bert: 'Legal-BERT',
        naive_baseline: 'Naive Baseline',
        gradient_boosting: 'Gradient Boosting',
        distilbert: 'DistilBERT'
    };

    if (data.predictions && Object.keys(data.predictions).length) {
        pEl.innerHTML = `<table class="predictions-table">
            <thead><tr>
                <th>Model</th><th style="text-align:right">Predicted</th><th style="text-align:right">Actual</th>
            </tr></thead>
            <tbody>${Object.entries(data.predictions).map(([m, pred]) => {
                const actual = data.consequence_tier;
                const match = pred === actual;
                return `<tr>
                    <td>${mLabels[m] || m}</td>
                    <td style="text-align:right; color:${match ? '#4ecdc4' : '#ff3333'}">${tierText[pred] ?? pred}</td>
                    <td style="text-align:right">${tierText[actual] ?? actual}</td>
                </tr>`;
            }).join('')}</tbody></table>`;
    } else {
        pEl.innerHTML = '<p style="color:#555; font-size:0.85rem;">No prediction data available.</p>';
    }

    // Connections
    const connEl = document.getElementById('modal-connections');
    if (data.connections && data.connections.length) {
        connEl.innerHTML = data.connections.map(c => {
            const safe = c.name.replace(/'/g, "\\'");
            return `<span class="person-pill" onclick="openPersonModal('${safe}')"
                         title="${c.weight} shared documents">
                ${c.name} <small style="opacity:0.5">(${c.weight})</small>
            </span>`;
        }).join('');
    } else {
        connEl.innerHTML = '<p style="color:#555; font-size:0.85rem;">No connections found in documents.</p>';
    }

    // Summary (lazy-loaded)
    const summarySection = document.getElementById('modal-summary-section');
    const summaryText = document.getElementById('modal-summary-text');
    const citationsEl = document.getElementById('modal-citations');

    if (data.has_summary) {
        summarySection.style.display = '';
        summaryText.textContent = 'Loading summary...';
        citationsEl.innerHTML = '';

        fetch(`/api/person/${encodeURIComponent(data.name)}/summary`)
            .then(r => r.json())
            .then(summary => {
                summaryText.textContent = summary.summary_text || 'No summary available.';
                if (summary.citations && summary.citations.length) {
                    citationsEl.innerHTML = summary.citations.slice(0, 10).map(c => `
                        <div class="citation-item">
                            <span class="citation-type">${c.doc_type || 'document'}</span>
                            <span class="citation-id">${c.doc_id || ''}</span>
                            ${c.date ? `<span class="citation-date">${c.date}</span>` : ''}
                            <p class="citation-snippet">${c.snippet || ''}</p>
                            ${c.jmail_url ? `<a href="${c.jmail_url}" target="_blank" rel="noopener noreferrer" class="citation-link">View on jmail.world &rarr;</a>` : ''}
                        </div>
                    `).join('');
                }
            })
            .catch(() => {
                summaryText.textContent = 'Summary could not be loaded.';
            });
    } else {
        summarySection.style.display = 'none';
    }

    // Show
    modal.removeAttribute('hidden');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function createModalGauge(score) {
    const level = getSeverityLevel(score);
    const gaugeData = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: score,
        number: { suffix: ' / 10', font: { size: 24, color: '#e8e8e8' } },
        gauge: {
            axis: { range: [0, 10], tickcolor: '#333', dtick: 2 },
            bar: { color: SEVERITY_COLORS[level] || '#555' },
            bgcolor: '#1a1a1a',
            borderwidth: 0,
            steps: [
                { range: [0, 1],   color: 'rgba(85,85,85,0.15)' },
                { range: [1, 4],   color: 'rgba(78,205,196,0.1)' },
                { range: [4, 7],   color: 'rgba(255,193,7,0.1)' },
                { range: [7, 8.5], color: 'rgba(255,107,53,0.1)' },
                { range: [8.5, 10], color: 'rgba(255,51,51,0.1)' }
            ]
        }
    }];

    Plotly.newPlot('modal-gauge', gaugeData, {
        ...PLOTLY_DARK,
        height: 170,
        margin: { t: 10, b: 0, l: 20, r: 20 }
    }, { displayModeBar: false, responsive: true });
}

function getSeverityLevel(score) {
    if (score >= 8.5) return 'Critical';
    if (score >= 7.0) return 'High';
    if (score >= 4.0) return 'Medium';
    if (score >= 1.0) return 'Low';
    return 'Minimal';
}

function closeModal() {
    const modal = document.getElementById('person-modal');
    if (!modal) return;
    modal.classList.remove('active');
    modal.setAttribute('hidden', '');
    document.body.style.overflow = '';
}

function setupModal() {
    const modal = document.getElementById('person-modal');
    if (!modal) return;

    const closeBtn = modal.querySelector('.modal-close');
    if (closeBtn) closeBtn.addEventListener('click', closeModal);

    modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}

/* ============================================================
   6. ANALYSIS CHARTS (Plotly — dark palette)
   ============================================================ */

function createScatterPlot() {
    if (!STATE.chartData.length) return;

    const tiers  = ['Low', 'Medium', 'High', 'Very High'];
    const colors = ['#4ecdc4', '#ffc107', '#ff6b35', '#ff3333'];

    const traces = tiers.map((tier, i) => {
        const td = STATE.chartData.filter(d => d.power_tier === tier);
        return {
            x: td.map(d => d.severity_score),
            y: td.map(d => d.consequence_tier + (Math.random() - 0.5) * 0.12),
            mode: 'markers', type: 'scatter', name: tier,
            marker: { size: 10, color: colors[i], opacity: 0.8, line: { color: '#000', width: 1 } },
            text: td.map(d => d.name),
            hovertemplate: '<b>%{text}</b><br>Severity: %{x:.2f}<br>Consequence: %{y:.0f}<extra></extra>'
        };
    });

    Plotly.newPlot('scatterPlot', traces, {
        ...PLOTLY_DARK,
        xaxis: { title: 'Severity Score', color: '#888', gridcolor: '#1e1e1e', zeroline: false },
        yaxis: { title: 'Consequence Tier', color: '#888', gridcolor: '#1e1e1e', zeroline: false,
                 tickvals: [0, 1], ticktext: ['None', 'Consequence'] },
        height: 420,
        margin: { t: 20, b: 50, l: 60, r: 20 },
        hovermode: 'closest',
        legend: { bgcolor: 'transparent', font: { size: 10, color: '#888' } }
    }, { responsive: true, displaylogo: false });
}

function displayModelResults() {
    const data = STATE.modelResults;
    if (!data || data.error) return;
    const container = document.getElementById('modelPerformance');
    if (!container) return;

    const order  = ['logistic_baseline', 'random_forest_tfidf', 'legal_bert', 'naive_baseline', 'gradient_boosting', 'distilbert'];
    const labels = {
        logistic_baseline: 'Logistic Regression',
        random_forest_tfidf: 'Random Forest + TF-IDF',
        legal_bert: 'Legal-BERT',
        naive_baseline: 'Naive Baseline',
        gradient_boosting: 'Gradient Boosting',
        distilbert: 'DistilBERT'
    };

    container.innerHTML = order.filter(m => data[m]).map(m => {
        const met = data[m];
        const acc = (met.accuracy * 100).toFixed(1);
        const f1  = (met.f1_macro * 100).toFixed(1);
        const best = m === 'random_forest_tfidf' || m === 'gradient_boosting';

        return `<div class="model-row${best ? ' best' : ''}">
            <div class="model-row-header">
                <span class="model-label">${labels[m]}</span>
                <span class="model-acc">${acc}%</span>
            </div>
            <div class="model-row-sub">Accuracy ${acc}% &middot; F1 ${f1}%</div>
        </div>`;
    }).join('');
}

function createExperimentChart() {
    if (!STATE.experimentResults.length) return;
    const data = STATE.experimentResults;

    const colors = data.map(d =>
        d.correlation > 0.5 ? '#4ecdc4' : d.correlation > 0.3 ? '#ffc107' : '#ff3333'
    );

    Plotly.newPlot('experimentChart', [{
        x: data.map(d => d.power_tier),
        y: data.map(d => d.correlation),
        type: 'bar',
        marker: { color: colors, line: { color: '#000', width: 1 } },
        text: data.map(d => d.correlation.toFixed(3)),
        textposition: 'outside',
        textfont: { color: '#888', size: 10 },
        hovertemplate: '<b>%{x}</b><br>Correlation: %{y:.3f}<br>N: %{customdata}<extra></extra>',
        customdata: data.map(d => d.n_people)
    }], {
        ...PLOTLY_DARK,
        xaxis: { color: '#888' },
        yaxis: { title: 'Correlation', color: '#888', gridcolor: '#1e1e1e', range: [-0.2, 1.0] },
        height: 320,
        margin: { t: 20, b: 50, l: 50, r: 20 },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function createAblationChart() {
    if (!STATE.ablationResults.length) return;
    const data = STATE.ablationResults;

    const labelMap = {
        'All Features': 'All Features',
        'Severity Score Only': 'Severity Only',
        'NLP Features Only': 'NLP Only',
        'Without mention_count': '\u2212 Documents',
        'Without total_mentions': '\u2212 Total Mentions',
        'Without mean_context_sentiment': '\u2212 Sentiment',
        'Without cooccurrence_score': '\u2212 Co-occurrence',
        'Without doc_type_diversity': '\u2212 Doc Types',
        'Without name_in_subject_line': '\u2212 In Subject',
        'Without severity_score': '\u2212 Severity Score'
    };

    const labels = data.map(d => labelMap[d.run] || d.run);
    const f1s    = data.map(d => d.f1_macro);
    const baseF1 = f1s[0] || 0;

    const colors = data.map(d => {
        if (d.run === 'All Features') return '#4ecdc4';
        if (d.run.includes('Only')) return '#ffc107';
        return d.f1_macro < baseF1 ? '#ff3333' : d.f1_macro > baseF1 ? '#4ecdc4' : '#888';
    });

    Plotly.newPlot('ablationChart', [
        {
            x: labels, y: f1s, type: 'bar',
            marker: { color: colors, line: { color: '#000', width: 1 } },
            text: f1s.map(v => v.toFixed(3)),
            textposition: 'outside',
            textfont: { color: '#888', size: 10 },
            hovertemplate: '<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>'
        },
        {
            x: labels, y: Array(labels.length).fill(baseF1),
            type: 'scatter', mode: 'lines',
            line: { color: 'rgba(255,255,255,0.25)', width: 1.5, dash: 'dash' },
            showlegend: false, hoverinfo: 'skip'
        }
    ], {
        ...PLOTLY_DARK,
        xaxis: { color: '#888', tickangle: -35 },
        yaxis: { title: 'F1 Score (Macro)', color: '#888', gridcolor: '#1e1e1e', range: [0, 1.05] },
        height: 400,
        margin: { t: 20, b: 110, l: 50, r: 20 },
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}

/* ============================================================
   7. SIDE NAVIGATION (IntersectionObserver)
   ============================================================ */

function setupSideNav() {
    const sections = document.querySelectorAll('section[id]');
    const dots     = document.querySelectorAll('.nav-dot');
    if (!sections.length || !dots.length) return;

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                dots.forEach(d => d.classList.remove('active'));
                const active = document.querySelector(`.nav-dot[href="#${entry.target.id}"]`);
                if (active) active.classList.add('active');
            }
        });
    }, { threshold: 0.3, rootMargin: '-10% 0px -10% 0px' });

    sections.forEach(s => observer.observe(s));
}

/* ============================================================
   8. GSAP SCROLL ANIMATIONS
   ============================================================ */

function setupAnimations() {
    if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;
    gsap.registerPlugin(ScrollTrigger);

    // Hero entrance
    gsap.from('.hero-title',    { opacity: 0, y: 40, duration: 1.2, ease: 'power3.out' });
    gsap.from('.hero-subtitle', { opacity: 0, y: 30, duration: 1,   delay: 0.3, ease: 'power3.out' });
    gsap.from('.hero-tagline',  { opacity: 0, y: 20, duration: 0.8, delay: 0.5, ease: 'power3.out' });

    // Section headers
    gsap.utils.toArray('.section-header').forEach(el => {
        gsap.from(el, {
            opacity: 0, y: 40, duration: 0.8, ease: 'power2.out',
            scrollTrigger: { trigger: el, start: 'top 82%', toggleActions: 'play none none none' }
        });
    });

    // Analysis panels stagger
    gsap.utils.toArray('.analysis-panel').forEach((el, i) => {
        gsap.from(el, {
            opacity: 0, y: 30, duration: 0.6, delay: i * 0.12, ease: 'power2.out',
            scrollTrigger: { trigger: el, start: 'top 85%', toggleActions: 'play none none none' }
        });
    });
}

/* ============================================================
   9. INITIALIZATION
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing The Accountability Gap...');

    setupPeopleControls();
    setupModal();
    setupSideNav();
    setupAnimations();
    loadAllData();
});

// Debounced resize for network graph
let _resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => {
        if (STATE.people.length) initNetworkGraph();
    }, 400);
});
