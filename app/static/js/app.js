// Attribution: Scaffolded with AI assistance (Claude, Anthropic)

/**
 * The Accountability Index — Main Application JavaScript
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
    activeSort: 'accountability',
    searchQuery: ''
};

const LEVEL_COLORS = {
    Critical: '#ef4444',
    High:     '#f59e0b',
    Moderate: '#3b82f6',
    Low:      '#06b6d4',
    Minimal:  '#475569'
};

const PLOTLY_THEME = {
    paper_bgcolor: '#111827',
    plot_bgcolor:  '#111827',
    font: { color: '#94a3b8', family: 'DM Sans, sans-serif', size: 11 }
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

        initNetworkGraph();
        renderPeopleGrid();
        renderModelCards();
        createMetricsBarChart();
        createConfusionMatrices();
        createAblationChart();
        createExperimentChart();
        renderLimitations();
        createScatterPlot();

    } catch (err) {
        console.error('Failed to load data:', err);
    }
}

/* ============================================================
   3. ANIMATED STAT COUNTERS
   ============================================================ */

function initStatCounters() {
    const cards = document.querySelectorAll('.stat-card');
    if (!cards.length) return;

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    cards.forEach(c => observer.observe(c));
}

function animateCounter(card) {
    const target = parseInt(card.dataset.count) || 0;
    const suffix = card.dataset.suffix || '';
    const el = card.querySelector('.stat-number');
    if (!el) return;

    const duration = 1500;
    const start = performance.now();

    function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const val = Math.round(eased * target);
        el.textContent = val.toLocaleString() + suffix;
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

/* ============================================================
   4. D3.js NETWORK GRAPH
   ============================================================ */

function initNetworkGraph() {
    const container = document.getElementById('network-graph');
    if (!container || !STATE.people.length) return;

    const width = container.clientWidth || 900;
    const height = container.clientHeight || Math.max(600, window.innerHeight * 0.7);
    container.innerHTML = '';

    const nameSet = new Set(STATE.people.map(p => p.name));

    const nodes = STATE.people.map(p => ({
        id:          p.name,
        score:       p.accountability_score,
        level:       p.level,
        consequence: p.consequence_tier,
        radius:      Math.max(10, Math.sqrt(Math.max(1, p.accountability_score)) * 5.5),
        image_url:   p.image_url || ''
    }));

    const links = STATE.edges
        .filter(e => nameSet.has(e.source) && nameSet.has(e.target))
        .map(e => ({ source: e.source, target: e.target, weight: e.weight }));

    const maxWeight = d3.max(links, d => d.weight) || 1;

    const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.25, 5]).on('zoom', e => g.attr('transform', e.transform)));

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(90).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-180))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.radius + 3));

    const link = g.append('g').selectAll('line').data(links).join('line')
        .attr('stroke', '#1e293b')
        .attr('stroke-opacity', d => 0.1 + (d.weight / maxWeight) * 0.5)
        .attr('stroke-width', d => 0.4 + (d.weight / maxWeight) * 3);

    const defs = svg.append('defs');
    nodes.forEach((d, i) => {
        defs.append('clipPath').attr('id', `nc-${i}`)
            .append('circle').attr('cx', 0).attr('cy', 0).attr('r', d.radius - 1.5);
    });

    const node = g.append('g').selectAll('g').data(nodes).join('g')
        .attr('cursor', 'pointer').call(makeDrag(simulation));

    node.append('circle')
        .attr('r', d => d.radius)
        .attr('fill', '#0f172a')
        .attr('stroke', d => LEVEL_COLORS[d.level] || '#475569')
        .attr('stroke-width', 2);

    node.append('image')
        .attr('href', d => d.image_url)
        .attr('x', d => -d.radius + 1.5).attr('y', d => -d.radius + 1.5)
        .attr('width', d => (d.radius - 1.5) * 2).attr('height', d => (d.radius - 1.5) * 2)
        .attr('clip-path', (d, i) => `url(#nc-${i})`)
        .attr('preserveAspectRatio', 'xMidYMid slice');

    const label = g.append('g').selectAll('text').data(nodes).join('text')
        .text(d => d.id)
        .attr('font-size', d => Math.max(8, d.radius * 0.7))
        .attr('fill', '#94a3b8').attr('text-anchor', 'middle')
        .attr('dy', d => d.radius + 12)
        .attr('pointer-events', 'none')
        .style('font-family', 'DM Sans, sans-serif')
        .style('opacity', d => d.score >= 4 ? 1 : 0);

    node.on('mouseover', function (event, d) {
        const connected = new Set([d.id]);
        links.forEach(l => {
            const s = typeof l.source === 'object' ? l.source.id : l.source;
            const t = typeof l.target === 'object' ? l.target.id : l.target;
            if (s === d.id) connected.add(t);
            if (t === d.id) connected.add(s);
        });
        node.attr('opacity', n => connected.has(n.__data__.id) ? 1 : 0.1);
        link.attr('opacity', l => {
            const s = typeof l.source === 'object' ? l.source.id : l.source;
            const t = typeof l.target === 'object' ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? 0.85 : 0.02;
        });
        label.style('opacity', n => connected.has(n.id) ? 1 : 0);
        d3.select(this).select('circle').attr('stroke', '#fff').attr('stroke-width', 3);
    })
    .on('mouseout', function () {
        node.attr('opacity', 1);
        link.attr('stroke-opacity', d => 0.1 + (d.weight / maxWeight) * 0.5).attr('opacity', null);
        label.style('opacity', d => d.score >= 4 ? 1 : 0);
        d3.select(this).select('circle').attr('stroke', d => LEVEL_COLORS[d.level] || '#475569').attr('stroke-width', 2);
    })
    .on('click', (event, d) => openPersonModal(d.id));

    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
        label.attr('x', d => d.x).attr('y', d => d.y);
    });
}

function makeDrag(sim) {
    return d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });
}

/* ============================================================
   5. PEOPLE GRID
   ============================================================ */

function renderPeopleGrid() {
    const container = document.getElementById('people-grid');
    if (!container) return;

    let list = [...STATE.people];

    if (STATE.searchQuery) {
        const q = STATE.searchQuery.toLowerCase();
        list = list.filter(p => p.name.toLowerCase().includes(q));
    }

    if (STATE.activeFilter !== 'all') {
        list = list.filter(p => p.level.toLowerCase() === STATE.activeFilter);
    }

    switch (STATE.activeSort) {
        case 'accountability':
            list.sort((a, b) => b.accountability_score - a.accountability_score);
            break;
        case 'name':
            list.sort((a, b) => a.name.localeCompare(b.name));
            break;
        case 'consequence':
            list.sort((a, b) => (b.consequence_tier - a.consequence_tier) || (b.accountability_score - a.accountability_score));
            break;
        case 'evidence':
            list.sort((a, b) => b.evidence_index - a.evidence_index);
            break;
    }

    if (!list.length) {
        container.innerHTML = '<p style="text-align:center;color:#475569;grid-column:1/-1;padding:3rem;">No individuals match your criteria.</p>';
        return;
    }

    container.innerHTML = list.map(p => {
        const lvl = p.level.toLowerCase();
        const color = LEVEL_COLORS[p.level] || '#475569';
        const cLabel = p.badge ? p.badge.label : '';
        const safeName = p.name.replace(/'/g, "\\'");
        const imgUrl = p.image_url || '/static/images/people/placeholder.png';

        return `
            <div class="person-card level-${lvl}" role="listitem"
                 onclick="openPersonModal('${safeName}')" tabindex="0"
                 aria-label="${p.name}, accountability ${p.accountability_score.toFixed(1)}">
                <img src="${imgUrl}" alt="" class="card-avatar"
                     onerror="this.style.display='none'" loading="lazy" />
                <div class="name">${p.name}</div>
                <div class="score" style="color:${color}">${p.accountability_score.toFixed(1)}</div>
                <span class="level-badge" style="color:${color}">${p.level}</span>
                <span class="consequence-badge">${cLabel}</span>
            </div>`;
    }).join('');
}

function setupPeopleControls() {
    const searchInput = document.getElementById('personSearch');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            STATE.searchQuery = searchInput.value.trim();
            renderPeopleGrid();
        });
    }

    document.querySelectorAll('.pill[data-filter]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            STATE.activeFilter = btn.dataset.filter;
            renderPeopleGrid();
        });
    });

    const sortSelect = document.getElementById('sortSelect');
    if (sortSelect) {
        sortSelect.addEventListener('change', () => {
            STATE.activeSort = sortSelect.value;
            renderPeopleGrid();
        });
    }
}

/* ============================================================
   6. PERSON MODAL
   ============================================================ */

function openPersonModal(name) {
    fetch(`/api/person/${encodeURIComponent(name)}`)
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            populateModal(data);
        })
        .catch(err => console.error('Error loading person:', err));
}

function populateModal(data) {
    const modal = document.getElementById('person-modal');
    if (!modal) return;

    const imgEl = document.getElementById('modal-image');
    if (imgEl) {
        imgEl.src = data.image_url || '/static/images/people/placeholder.png';
        imgEl.alt = data.name;
        imgEl.onerror = function() { this.src = '/static/images/people/placeholder.png'; };
    }

    document.getElementById('modal-name').textContent = data.name;
    const badge = document.getElementById('modal-level');
    badge.textContent = data.level;
    badge.style.color = LEVEL_COLORS[data.level] || '#475569';

    createModalGauge(data.accountability_score);

    const cEl = document.getElementById('modal-consequence');
    const cColor = data.badge?.color === 'hard' ? '#ef4444' : data.badge?.color === 'soft' ? '#f59e0b' : '#475569';
    cEl.innerHTML = `
        <div style="margin-bottom:0.5rem;">
            <span style="font-size:0.82rem;padding:0.25rem 0.7rem;border-radius:4px;background:rgba(255,255,255,0.05);color:${cColor};">${data.badge?.label || 'Unknown'}</span>
        </div>
        <p style="font-size:0.85rem;color:#94a3b8;line-height:1.65;">${data.consequence_description || 'No information available'}</p>`;

    const fEl = document.getElementById('modal-features');
    const fLabels = {
        mention_count: 'Documents', total_mentions: 'Total Mentions',
        mean_sentiment: 'Avg Sentiment', cooccurrence_score: 'Co-occurrence',
        doc_type_diversity: 'Doc Types', in_subject_line: 'In Subject'
    };
    if (data.features) {
        fEl.innerHTML = Object.entries(data.features).map(([k, v]) => {
            let display = v;
            if (typeof v === 'boolean') display = v ? 'Yes' : 'No';
            else if (typeof v === 'number') display = Number.isInteger(v) ? v : v.toFixed(3);
            return `<div class="feature-item"><div class="feature-label">${fLabels[k] || k}</div><div class="feature-value">${display}</div></div>`;
        }).join('');
    }

    const pEl = document.getElementById('modal-predictions');
    const tierText = { 0: 'No Consequence', 1: 'Consequence' };
    const mLabels = {
        logistic_baseline: 'Logistic Regression',
        random_forest_tfidf: 'Random Forest + TF-IDF',
        legal_bert: 'Legal-BERT'
    };
    if (data.predictions && Object.keys(data.predictions).length) {
        pEl.innerHTML = `<table class="predictions-table">
            <thead><tr><th>Model</th><th style="text-align:right">Predicted</th><th style="text-align:right">Actual</th></tr></thead>
            <tbody>${Object.entries(data.predictions).map(([m, pred]) => {
                const actual = data.consequence_tier > 0 ? 1 : 0;
                const match = pred === actual;
                return `<tr><td>${mLabels[m] || m}</td>
                    <td style="text-align:right;color:${match ? '#10b981' : '#ef4444'}">${tierText[pred] ?? pred}</td>
                    <td style="text-align:right">${tierText[actual] ?? actual}</td></tr>`;
            }).join('')}</tbody></table>`;
    } else {
        pEl.innerHTML = '<p style="color:#475569;font-size:0.82rem;">No prediction data for this individual (not in test set).</p>';
    }

    const connEl = document.getElementById('modal-connections');
    if (data.connections && data.connections.length) {
        connEl.innerHTML = data.connections.map(c => {
            const safe = c.name.replace(/'/g, "\\'");
            return `<span class="person-pill" onclick="openPersonModal('${safe}')" title="${c.weight} shared documents">${c.name} <small style="opacity:0.5">(${c.weight})</small></span>`;
        }).join('');
    } else {
        connEl.innerHTML = '<p style="color:#475569;font-size:0.82rem;">No co-occurrence connections found.</p>';
    }

    // Summary & Citations with real document links
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
                    citationsEl.innerHTML = summary.citations.slice(0, 10).map(c => {
                        const batesNumber = extractBatesNumber(c.doc_id);
                        const docUrl = buildDocumentUrl(batesNumber, c.source_volume);
                        return `
                            <div class="citation-item">
                                <span class="citation-type">${c.doc_type || 'document'}</span>
                                <span class="citation-id">${batesNumber || c.doc_id}</span>
                                <p class="citation-snippet">${c.snippet || ''}</p>
                                ${docUrl ? `<a href="${docUrl}" target="_blank" rel="noopener noreferrer" class="citation-link">Search Document Archives &rarr;</a>` : `<span style="font-size:0.68rem;color:#64748b;">Source: DOJ Release ${c.source_volume || 'unknown'}</span>`}
                            </div>`;
                    }).join('');
                }
            })
            .catch(() => { summaryText.textContent = 'Summary could not be loaded.'; });
    } else {
        summarySection.style.display = 'none';
    }

    modal.removeAttribute('hidden');
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function extractBatesNumber(docId) {
    if (!docId) return '';
    const parts = docId.split('_');
    if (parts.length >= 3) return parts.slice(2).join('_');
    return docId;
}

function buildDocumentUrl(batesNumber, sourceVolume) {
    if (!batesNumber) return '';
    return `https://www.documentcloud.org/documents/search/q:${encodeURIComponent(batesNumber)}`;
}

function createModalGauge(score) {
    const level = getLevel(score);
    Plotly.newPlot('modal-gauge', [{
        type: 'indicator', mode: 'gauge+number', value: score,
        number: { suffix: ' / 10', font: { size: 22, color: '#f8fafc' } },
        gauge: {
            axis: { range: [0, 10], tickcolor: '#1e293b', dtick: 2 },
            bar: { color: LEVEL_COLORS[level] || '#475569' },
            bgcolor: '#0f172a', borderwidth: 0,
            steps: [
                { range: [0, 1],   color: 'rgba(71,85,105,0.15)' },
                { range: [1, 2.5], color: 'rgba(6,182,212,0.08)' },
                { range: [2.5, 5], color: 'rgba(59,130,246,0.08)' },
                { range: [5, 7.5], color: 'rgba(245,158,11,0.08)' },
                { range: [7.5, 10], color: 'rgba(239,68,68,0.08)' }
            ]
        }
    }], { ...PLOTLY_THEME, height: 165, margin: { t: 10, b: 0, l: 20, r: 20 } },
    { displayModeBar: false, responsive: true });
}

function getLevel(score) {
    if (score >= 7.5) return 'Critical';
    if (score >= 5.0) return 'High';
    if (score >= 2.5) return 'Moderate';
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
    modal.querySelector('.modal-close')?.addEventListener('click', closeModal);
    modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}

/* ============================================================
   7. MODELS & EVALUATION
   ============================================================ */

function setupModelTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = document.getElementById(`tab-${tab.dataset.tab}`);
            if (target) target.classList.add('active');
        });
    });
}

function renderModelCards() {
    const data = STATE.modelResults;
    if (!data || data.error) return;
    const container = document.getElementById('modelCards');
    if (!container) return;

    const models = ['logistic_baseline', 'random_forest_tfidf', 'legal_bert'];
    const labels = {
        logistic_baseline: 'Logistic Regression',
        random_forest_tfidf: 'Random Forest + TF-IDF',
        legal_bert: 'Legal-BERT (Fine-tuned)'
    };

    container.innerHTML = models.filter(m => data[m]).map(m => {
        const met = data[m];
        const acc = (met.accuracy * 100).toFixed(1);
        const f1 = (met.f1_macro * 100).toFixed(1);
        const mcc = (met.mcc || 0).toFixed(3);
        const prec = ((met.precision_positive || 0) * 100).toFixed(1);
        const rec = ((met.recall_positive || 0) * 100).toFixed(1);
        const best = m === 'random_forest_tfidf';

        return `<div class="model-card${best ? ' best' : ''}">
            <div class="model-card-header">
                <span class="model-card-name">${labels[m]}${best ? ' (Best)' : ''}</span>
                <span class="model-card-acc">${acc}%</span>
            </div>
            <div class="model-card-desc">${met.description || ''}</div>
            <div class="model-metrics-row">
                <span class="metric-chip">F1 <span class="metric-val">${f1}%</span></span>
                <span class="metric-chip">MCC <span class="metric-val">${mcc}</span></span>
                <span class="metric-chip">Precision <span class="metric-val">${prec}%</span></span>
                <span class="metric-chip">Recall <span class="metric-val">${rec}%</span></span>
            </div>
        </div>`;
    }).join('');
}

function createMetricsBarChart() {
    const data = STATE.modelResults;
    if (!data || data.error) return;

    const models = ['logistic_baseline', 'random_forest_tfidf', 'legal_bert'].filter(m => data[m]);
    const labels = { logistic_baseline: 'Logistic', random_forest_tfidf: 'RF+TF-IDF', legal_bert: 'Legal-BERT' };

    const metrics = ['accuracy', 'f1_macro', 'mcc', 'precision_positive', 'recall_positive'];
    const metricLabels = { accuracy: 'Accuracy', f1_macro: 'F1 Macro', mcc: 'MCC', precision_positive: 'Precision', recall_positive: 'Recall' };
    const colors = ['#3b82f6', '#06b6d4', '#6366f1', '#f59e0b', '#10b981'];

    const traces = metrics.map((metric, i) => ({
        x: models.map(m => labels[m]),
        y: models.map(m => data[m][metric] || 0),
        name: metricLabels[metric],
        type: 'bar',
        marker: { color: colors[i], line: { color: '#0f172a', width: 1 } }
    }));

    Plotly.newPlot('metricsBarChart', traces, {
        ...PLOTLY_THEME,
        barmode: 'group',
        xaxis: { color: '#94a3b8' },
        yaxis: { title: 'Score', color: '#94a3b8', gridcolor: '#1e293b', range: [0, 1.05] },
        height: 380,
        margin: { t: 20, b: 50, l: 50, r: 20 },
        legend: { bgcolor: 'transparent', font: { size: 10, color: '#94a3b8' }, orientation: 'h', y: -0.2 }
    }, { responsive: true, displaylogo: false });
}

function createConfusionMatrices() {
    const data = STATE.modelResults;
    if (!data || data.error) return;
    const container = document.getElementById('confusionMatrices');
    if (!container) return;

    const models = ['logistic_baseline', 'random_forest_tfidf', 'legal_bert'].filter(m => data[m] && data[m].confusion_matrix);
    const labels = { logistic_baseline: 'Logistic Regression', random_forest_tfidf: 'Random Forest + TF-IDF', legal_bert: 'Legal-BERT' };

    container.innerHTML = models.map(m => {
        return `<div class="confusion-panel">
            <h4>${labels[m]}</h4>
            <div id="cm-${m}" style="width:100%;height:280px;"></div>
        </div>`;
    }).join('');

    models.forEach(m => {
        const cm = data[m].confusion_matrix;
        const z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]];

        Plotly.newPlot(`cm-${m}`, [{
            z: z, type: 'heatmap',
            colorscale: [[0, '#0f172a'], [0.5, '#1e40af'], [1, '#3b82f6']],
            showscale: false, xgap: 2, ygap: 2,
            hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        }], {
            ...PLOTLY_THEME,
            xaxis: { title: 'Predicted', tickvals: [0, 1], ticktext: ['No Cons.', 'Cons.'], color: '#94a3b8', side: 'bottom' },
            yaxis: { title: 'Actual', tickvals: [0, 1], ticktext: ['No Cons.', 'Cons.'], color: '#94a3b8', autorange: 'reversed' },
            height: 260, margin: { t: 10, b: 50, l: 60, r: 10 },
            annotations: z.flatMap((row, i) => row.map((val, j) => ({
                x: j, y: i, text: String(val), showarrow: false,
                font: { color: val > 10 ? '#fff' : '#e2e8f0', size: 16, family: 'Space Grotesk' }
            })))
        }, { displayModeBar: false, responsive: true });
    });
}

function createAblationChart() {
    if (!STATE.ablationResults.length) return;
    const data = STATE.ablationResults;

    const labelMap = {
        'All Features': 'All Features', 'Severity Score Only': 'Severity Only',
        'NLP Features Only': 'NLP Only', 'Without mention_count': '\u2212 Documents',
        'Without total_mentions': '\u2212 Total Mentions', 'Without mean_context_sentiment': '\u2212 Sentiment',
        'Without cooccurrence_score': '\u2212 Co-occurrence', 'Without doc_type_diversity': '\u2212 Doc Types',
        'Without name_in_subject_line': '\u2212 In Subject', 'Without severity_score': '\u2212 Severity Score',
        'TF-IDF Only (no tabular)': 'TF-IDF Only', 'Tabular + TF-IDF Combined': 'Tabular+TF-IDF'
    };

    const labels = data.map(d => labelMap[d.run] || d.run);
    const f1s = data.map(d => d.f1_macro);
    const baseF1 = f1s[0] || 0;

    const colors = data.map(d => {
        if (d.run === 'All Features') return '#3b82f6';
        if (d.run.includes('Only') || d.run.includes('Combined')) return '#06b6d4';
        return d.f1_macro < baseF1 ? '#ef4444' : '#10b981';
    });

    Plotly.newPlot('ablationChart', [
        {
            x: labels, y: f1s, type: 'bar',
            marker: { color: colors, line: { color: '#0f172a', width: 1 } },
            text: f1s.map(v => v.toFixed(3)), textposition: 'outside',
            textfont: { color: '#94a3b8', size: 10 },
            hovertemplate: '<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>'
        },
        {
            x: labels, y: Array(labels.length).fill(baseF1),
            type: 'scatter', mode: 'lines',
            line: { color: 'rgba(59,130,246,0.4)', width: 1.5, dash: 'dash' },
            showlegend: false, hoverinfo: 'skip'
        }
    ], {
        ...PLOTLY_THEME,
        xaxis: { color: '#94a3b8', tickangle: -35 },
        yaxis: { title: 'F1 Score (Macro)', color: '#94a3b8', gridcolor: '#1e293b', range: [0, 1.05] },
        height: 380, margin: { t: 20, b: 110, l: 50, r: 20 }, showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function createExperimentChart() {
    if (!STATE.experimentResults.length) return;
    const data = STATE.experimentResults;

    const colors = data.map(d =>
        d.correlation > 0.3 ? '#10b981' : d.correlation > 0 ? '#f59e0b' : '#ef4444'
    );

    Plotly.newPlot('experimentChart', [{
        x: data.map(d => d.power_tier),
        y: data.map(d => d.correlation),
        type: 'bar',
        marker: { color: colors, line: { color: '#0f172a', width: 1 } },
        text: data.map(d => d.correlation.toFixed(3)),
        textposition: 'outside',
        textfont: { color: '#94a3b8', size: 10 },
        hovertemplate: '<b>%{x}</b><br>Correlation: %{y:.3f}<br>N=%{customdata}<extra></extra>',
        customdata: data.map(d => d.n_people)
    }], {
        ...PLOTLY_THEME,
        xaxis: { color: '#94a3b8' },
        yaxis: { title: 'Correlation', color: '#94a3b8', gridcolor: '#1e293b', range: [-0.4, 0.5] },
        height: 340, margin: { t: 20, b: 50, l: 50, r: 20 }, showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function renderLimitations() {
    const container = document.getElementById('limitationsContent');
    if (!container) return;

    const st = STATE.modelResults?.stress_test || {};

    container.innerHTML = `
        <div class="limitation-card">
            <h4>Class Imbalance (${st.imbalance_ratio || 3.4}:1 ratio)</h4>
            <p>${st.class_distribution ? st.class_distribution.no_consequence : 51} individuals face no consequences vs only ${st.class_distribution ? st.class_distribution.has_consequence : 15} with consequences. A naive majority-class baseline achieves ${((st.baseline_majority_accuracy || 0.773) * 100).toFixed(1)}% accuracy.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>SMOTE oversampling or collecting more positive examples through expanded document sources.</p>
            </div>
        </div>
        <div class="limitation-card">
            <h4>Small Sample Size (66 individuals)</h4>
            <p>${st.cross_val_note || 'With 66 total samples, cross-validation folds have very few positive examples.'} Single misclassifications cause large metric swings.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>Document-level predictions (2,800+ samples) increase statistical power, as shown by Legal-BERT's approach.</p>
            </div>
        </div>
        <div class="limitation-card">
            <h4>Legal-BERT Underperformance (45.7% acc)</h4>
            <p>Despite domain-specific pretraining, Legal-BERT underperforms simpler models due to person-level aggregation noise and limited fine-tuning data.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>More fine-tuning data, attention-weighted person-level aggregation, and longer training could improve results.</p>
            </div>
        </div>
        <div class="limitation-card">
            <h4>Feature Dependency on Severity Score</h4>
            <p>Removing severity score drops F1 by ${Math.abs(st.ablation_severity_drop || -0.074).toFixed(3)}. NLP-only features achieve F1=${(st.ablation_nlp_only_f1 || 0.714).toFixed(3)}.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>Stronger NLP features (entity embeddings, relationship extraction, temporal analysis) could reduce external dependency.</p>
            </div>
        </div>`;
}

/* ============================================================
   8. ANALYSIS CHARTS
   ============================================================ */

function createScatterPlot() {
    if (!STATE.chartData.length) return;

    const tiers = ['Low', 'Moderate', 'High', 'Critical'];
    const colors = ['#06b6d4', '#3b82f6', '#f59e0b', '#ef4444'];

    const traces = tiers.map((tier, i) => {
        const td = STATE.chartData.filter(d => d.power_tier === tier);
        return {
            x: td.map(d => d.evidence_index),
            y: td.map(d => d.consequence_tier + (Math.random() - 0.5) * 0.15),
            mode: 'markers', type: 'scatter', name: tier,
            marker: { size: 10, color: colors[i], opacity: 0.85, line: { color: '#0f172a', width: 1 } },
            text: td.map(d => d.name),
            hovertemplate: '<b>%{text}</b><br>Evidence: %{x:.1f}<br>Consequence: %{y:.0f}<extra></extra>'
        };
    });

    Plotly.newPlot('scatterPlot', traces, {
        ...PLOTLY_THEME,
        xaxis: { title: 'Evidence Index (NLP-derived)', color: '#94a3b8', gridcolor: '#1e293b', zeroline: false },
        yaxis: { title: 'Consequence Tier', color: '#94a3b8', gridcolor: '#1e293b', zeroline: false,
                 tickvals: [0, 1, 2], ticktext: ['None', 'Soft', 'Hard'] },
        height: 420, margin: { t: 20, b: 60, l: 70, r: 20 },
        hovermode: 'closest',
        legend: { bgcolor: 'transparent', font: { size: 10, color: '#94a3b8' } }
    }, { responsive: true, displaylogo: false });
}

/* ============================================================
   9. SIDE NAVIGATION
   ============================================================ */

function setupSideNav() {
    const sections = document.querySelectorAll('section[id]');
    const dots = document.querySelectorAll('.nav-dot');
    if (!sections.length || !dots.length) return;

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                dots.forEach(d => d.classList.remove('active'));
                const active = document.querySelector(`.nav-dot[href="#${entry.target.id}"]`);
                if (active) active.classList.add('active');
            }
        });
    }, { threshold: 0.25, rootMargin: '-10% 0px -10% 0px' });

    sections.forEach(s => observer.observe(s));
}

/* ============================================================
   10. GSAP SCROLL ANIMATIONS
   ============================================================ */

function setupAnimations() {
    if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;
    gsap.registerPlugin(ScrollTrigger);

    gsap.from('.hero-badge', { opacity: 0, y: 20, duration: 0.8, ease: 'power3.out' });
    gsap.from('.hero-title', { opacity: 0, y: 40, duration: 1.2, delay: 0.15, ease: 'power3.out' });
    gsap.from('.hero-tagline', { opacity: 0, y: 20, duration: 0.8, delay: 0.4, ease: 'power3.out' });
    gsap.from('.hero-cta-row', { opacity: 0, y: 15, duration: 0.6, delay: 0.6, ease: 'power3.out' });

    gsap.utils.toArray('.section-header').forEach(el => {
        gsap.from(el, {
            opacity: 0, y: 30, duration: 0.8, ease: 'power2.out',
            scrollTrigger: { trigger: el, start: 'top 82%', toggleActions: 'play none none none' }
        });
    });

    gsap.utils.toArray('.approach-step').forEach((el, i) => {
        gsap.from(el, {
            opacity: 0, y: 20, duration: 0.5, delay: i * 0.1, ease: 'power2.out',
            scrollTrigger: { trigger: el, start: 'top 85%', toggleActions: 'play none none none' }
        });
    });

    gsap.utils.toArray('.stat-card').forEach((el, i) => {
        gsap.from(el, {
            opacity: 0, scale: 0.9, duration: 0.5, delay: i * 0.08, ease: 'back.out(1.5)',
            scrollTrigger: { trigger: el, start: 'top 88%', toggleActions: 'play none none none' }
        });
    });
}

/* ============================================================
   11. INITIALIZATION
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    setupPeopleControls();
    setupModal();
    setupModelTabs();
    setupSideNav();
    setupAnimations();
    initStatCounters();
    loadAllData();
});

let _resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(() => { if (STATE.people.length) initNetworkGraph(); }, 400);
});
