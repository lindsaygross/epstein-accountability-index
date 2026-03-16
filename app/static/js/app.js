// Project: The Impunity Index
// Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
// Advisor: Brinnae Bent, PhD
// Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
// Built with Claude AI assistance

/**
 * The Impunity Index — Main Application JavaScript
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
    activeSort: 'impunity',
    searchQuery: '',
    gridExpanded: false
};

const GRID_INITIAL_COUNT = 50;

const LEVEL_COLORS = {
    Critical: '#ef4444',
    High:     '#f59e0b',
    Moderate: '#3b82f6',
    Low:      '#06b6d4',
    Minimal:  '#475569'
};

function getPlotlyTheme() {
    const isLight = document.documentElement.dataset.theme === 'light';
    return {
        paper_bgcolor: isLight ? '#ffffff' : '#111827',
        plot_bgcolor:  isLight ? '#ffffff' : '#111827',
        font: { color: isLight ? '#475569' : '#94a3b8', family: 'DM Sans, sans-serif', size: 11 }
    };
}

// Default (will be overridden by getPlotlyTheme() in each chart call)
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
        createGeoMap();
        createSemanticSpace();

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

    // Build full link list first
    const allLinks = STATE.edges
        .filter(e => nameSet.has(e.source) && nameSet.has(e.target))
        .map(e => ({ source: e.source, target: e.target, weight: e.weight }));

    // Only show nodes that have at least one edge (connected nodes)
    const connectedNames = new Set();
    allLinks.forEach(l => { connectedNames.add(l.source); connectedNames.add(l.target); });

    const nodes = STATE.people
        .filter(p => connectedNames.has(p.name))
        .map(p => ({
            id:          p.name,
            score:       p.impunity_index,
            level:       p.level,
            consequence: p.consequence_tier,
            radius:      Math.max(8, Math.sqrt(Math.max(1, p.impunity_index)) * 5),
            image_url:   p.image_url || ''
        }));

    const links = allLinks;

    const maxWeight = d3.max(links, d => d.weight) || 1;

    const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.25, 5]).on('zoom', e => g.attr('transform', e.transform)));

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(90).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-180))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.radius + 3));

    const edgeColor = document.documentElement.dataset.theme === 'light' ? '#94a3b8' : '#475569';
    const link = g.append('g').selectAll('line').data(links).join('line')
        .attr('stroke', edgeColor)
        .attr('stroke-opacity', d => 0.3 + (d.weight / maxWeight) * 0.5)
        .attr('stroke-width', d => 0.8 + (d.weight / maxWeight) * 3);

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

    // Add a default person silhouette circle as fallback for missing images
    node.append('circle')
        .attr('r', d => d.radius * 0.55)
        .attr('fill', '#334155')
        .attr('class', 'node-fallback-icon');

    // Person silhouette SVG path (head + shoulders)
    node.append('path')
        .attr('d', d => {
            const s = d.radius * 0.45;
            return `M0,${-s*0.3} a${s*0.35},${s*0.35} 0 1,0 0.001,0 M${-s*0.6},${s*0.7} a${s*0.75},${s*0.6} 0 0,1 ${s*1.2},0`;
        })
        .attr('fill', '#64748b')
        .attr('stroke', 'none')
        .attr('class', 'node-fallback-icon');

    // Map node id to clip-path index
    const nodeIdToIdx = {};
    nodes.forEach((d, i) => { nodeIdToIdx[d.id] = i; });

    node.filter(d => d.image_url && d.image_url !== '/static/images/people/placeholder.png')
        .append('image')
        .attr('href', d => d.image_url)
        .attr('x', d => -d.radius + 1.5).attr('y', d => -d.radius + 1.5)
        .attr('width', d => (d.radius - 1.5) * 2).attr('height', d => (d.radius - 1.5) * 2)
        .attr('clip-path', d => `url(#nc-${nodeIdToIdx[d.id]})`)
        .attr('preserveAspectRatio', 'xMidYMid slice')
        .on('error', function() { d3.select(this).remove(); });

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
        link.attr('stroke-opacity', d => 0.3 + (d.weight / maxWeight) * 0.5).attr('opacity', null);
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
        case 'impunity':
            list.sort((a, b) => b.impunity_index - a.impunity_index);
            break;
        case 'name':
            list.sort((a, b) => a.name.localeCompare(b.name));
            break;
        case 'consequence':
            list.sort((a, b) => (b.consequence_tier - a.consequence_tier) || (b.impunity_index - a.impunity_index));
            break;
        case 'evidence':
            list.sort((a, b) => b.evidence_index - a.evidence_index);
            break;
    }

    if (!list.length) {
        container.innerHTML = '<p style="text-align:center;color:#475569;grid-column:1/-1;padding:3rem;">No individuals match your criteria.</p>';
        // Clear expand button
        const btn = document.getElementById('people-expand-btn');
        if (btn) btn.style.display = 'none';
        return;
    }

    const isSearchOrFilter = STATE.searchQuery || STATE.activeFilter !== 'all';
    const showAll = STATE.gridExpanded || isSearchOrFilter;
    const displayList = showAll ? list : list.slice(0, GRID_INITIAL_COUNT);

    function makeCard(p) {
        const lvl = p.level.toLowerCase();
        const color = LEVEL_COLORS[p.level] || '#475569';
        const cLabel = p.badge ? p.badge.label : '';
        const safeName = p.name.replace(/'/g, "\\'");
        const imgUrl = p.image_url || '/static/images/people/placeholder.png';
        return `
            <div class="person-card level-${lvl}" role="listitem"
                 onclick="openPersonModal('${safeName}')" tabindex="0"
                 aria-label="${p.name}, impunity ${p.impunity_index.toFixed(1)}">
                <img src="${imgUrl}" alt="" class="card-avatar"
                     onerror="this.style.display='none'" loading="lazy" />
                <div class="name">${p.name}</div>
                <div class="score" style="color:${color}">${p.impunity_index.toFixed(1)}</div>
                <span class="level-badge" style="color:${color}">${p.level}</span>
                <span class="consequence-badge">${cLabel}</span>
            </div>`;
    }

    container.innerHTML = displayList.map(makeCard).join('');

    // Update expand button
    const btn = document.getElementById('people-expand-btn');
    if (btn) {
        if (!isSearchOrFilter && list.length > GRID_INITIAL_COUNT) {
            btn.style.display = '';
            if (STATE.gridExpanded) {
                btn.textContent = `Collapse to top ${GRID_INITIAL_COUNT} ↑`;
            } else {
                btn.textContent = `Show all ${list.length.toLocaleString()} individuals ↓`;
            }
        } else {
            btn.style.display = 'none';
        }
    }
}

function setupPeopleControls() {
    const searchInput = document.getElementById('personSearch');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            STATE.searchQuery = searchInput.value.trim();
            STATE.gridExpanded = false;
            renderPeopleGrid();
        });
    }

    const expandBtn = document.getElementById('people-expand-btn');
    if (expandBtn) {
        expandBtn.addEventListener('click', () => {
            STATE.gridExpanded = !STATE.gridExpanded;
            renderPeopleGrid();
            if (!STATE.gridExpanded) {
                // Scroll back to top of people section
                document.getElementById('people')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
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

    // Bio line under name — nationality + description
    const bioEl = document.getElementById('modal-bio');
    if (bioEl) {
        const parts = [];
        if (data.nationality && data.nationality !== 'Unknown') parts.push(data.nationality);
        if (data.consequence_description) parts.push(data.consequence_description);
        bioEl.textContent = parts.join(' · ');
    }

    createModalGauge(data.impunity_index);

    // Show internal score breakdown
    const breakdownEl = document.getElementById('modal-score-breakdown');
    if (breakdownEl) {
        const modifier = data.consequence_tier === 0 ? 1.3 : data.consequence_tier === 1 ? 1.0 : 0.7;
        const modifierLabel = data.consequence_tier === 0 ? '×1.3 (no consequence)' : data.consequence_tier === 1 ? '×1.0 (soft consequence)' : '×0.7 (convicted)';
        breakdownEl.innerHTML = `
            <div class="breakdown-row">
                <span class="breakdown-label modal-tooltip-wrap">Evidence Index <span class="modal-info-icon">&#9432;<span class="modal-info-tooltip">A 0-10 score measuring how often and how prominently this person appears across the document corpus. Based on EFTA document count, DOJ mentions, keyword co-occurrence, flight logs, connections, and black book presence.</span></span></span>
                <span class="breakdown-value">${(data.evidence_index || 0).toFixed(1)}</span>
            </div>
            <div class="breakdown-row">
                <span class="breakdown-label modal-tooltip-wrap">Consequence Modifier <span class="modal-info-icon">&#9432;<span class="modal-info-tooltip">Adjusts the score based on real-world legal outcomes. No consequence: x1.3 (increases impunity). Soft consequence (resigned, sued): x1.0 (neutral). Hard consequence (convicted, imprisoned): x0.7 (reduces impunity).</span></span></span>
                <span class="breakdown-value">${modifierLabel}</span>
            </div>
            <div class="breakdown-formula">${(data.evidence_index || 0).toFixed(1)} × ${modifier} = <strong>${(data.impunity_index || 0).toFixed(1)}</strong></div>
            <div class="breakdown-explain">Higher impunity = more evidence, less consequence</div>`;
    }

    const cEl = document.getElementById('modal-consequence');
    const cColor = data.badge?.color === 'hard' ? '#ef4444' : data.badge?.color === 'soft' ? '#f59e0b' : '#475569';
    cEl.innerHTML = `<span style="font-size:0.85rem;padding:0.3rem 0.8rem;border-radius:4px;background:rgba(255,255,255,0.05);border:1px solid ${cColor}33;color:${cColor};font-weight:600;">${data.badge?.label || 'Unknown'}</span>`;

    // NLP Features — prefer ev_data fields (doc_mentions, keyword_cooccurrence, flights, connections, in_black_book)
    const fEl = document.getElementById('modal-features');
    if (fEl) {
        const ev = {
            docMentions: data.doc_mentions || (data.features && data.features.total_mentions) || 0,
            cooccurrence: data.keyword_cooccurrence || (data.features && data.features.cooccurrence_score) || 0,
            flights: data.flights || 0,
            connections: Array.isArray(data.connections) ? data.connections.length : (data.connections || 0),
            blackBook: data.in_black_book ? 'Yes' : 'No',
        };
        fEl.innerHTML = `
            <div class="feature-item"><div class="feature-label">EFTA Docs</div><div class="feature-value">${ev.docMentions.toLocaleString()}</div></div>
            <div class="feature-item"><div class="feature-label">Co-occurrence</div><div class="feature-value">${ev.cooccurrence.toLocaleString()}</div></div>
            <div class="feature-item"><div class="feature-label">Flight Legs</div><div class="feature-value">${ev.flights.toLocaleString()}</div></div>
            <div class="feature-item"><div class="feature-label">Connections</div><div class="feature-value">${ev.connections.toLocaleString()}</div></div>
            <div class="feature-item"><div class="feature-label">Black Book</div><div class="feature-value">${ev.blackBook}</div></div>`;
    }

    // ML Signals — show consensus probability bar (no predicted/actual, just signal strength)
    const pEl = document.getElementById('modal-predictions');
    if (pEl) {
        const mOrder = ['logistic_regression', 'random_forest_tfidf', 'sentence_transformer_svc'];
        const mLabels = {
            logistic_regression: 'Logistic Regression',
            random_forest_tfidf: 'Random Forest + TF-IDF',
            sentence_transformer_svc: 'ST + SVC (Semantic)',
        };
        if (data.predictions && Object.keys(data.predictions).length) {
            const mShortLabels = {
                logistic_regression: 'Log. Regression',
                random_forest_tfidf: 'RF + TF-IDF',
                sentence_transformer_svc: 'ST + SVC',
            };
            const bars = mOrder.filter(m => data.predictions[m] != null).map(m => {
                const pred = data.predictions[m];
                const prob = typeof pred === 'object' ? (pred.probability ?? 0) : 0;
                const pct = (prob * 100).toFixed(1);
                const color = prob >= 0.7 ? '#ef4444' : prob >= 0.4 ? '#f59e0b' : '#3b82f6';
                return `<div style="margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:0.2rem;">
                        <span style="color:var(--text-secondary)">${mShortLabels[m]}</span>
                        <span style="color:${color};font-weight:600">${pct}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07);border-radius:3px;height:6px;overflow:hidden;">
                        <div style="background:${color};width:${pct}%;height:100%;border-radius:3px;transition:width 0.6s ease;"></div>
                    </div>
                </div>`;
            });
            const consensus = data.predictions.consensus;
            const cProb = consensus ? (typeof consensus === 'object' ? (consensus.probability ?? 0) : 0) : 0;
            const cColor = cProb >= 0.7 ? '#ef4444' : cProb >= 0.4 ? '#f59e0b' : '#3b82f6';
            pEl.innerHTML = bars.join('') + (consensus != null ? `
                <div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid var(--border-subtle);">
                    <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:0.25rem;">
                        <span style="font-weight:600;color:var(--text-primary)">Consensus</span>
                        <span style="color:${cColor};font-weight:700">${(cProb * 100).toFixed(1)}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07);border-radius:3px;height:6px;overflow:hidden;">
                        <div style="background:${cColor};width:${(cProb*100).toFixed(1)}%;height:100%;border-radius:3px;"></div>
                    </div>
                    <p style="font-size:0.72rem;color:var(--text-secondary);margin-top:0.35rem;line-height:1.4;">Mean of 3 models — document evidence pattern, not a legal determination.</p>
                </div>` : '');
        } else {
            pEl.innerHTML = '<p style="color:var(--text-secondary);font-size:0.82rem;">No ML signal data available for this individual.</p>';
        }
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

    // Topic distribution badge row (if available)
    const fEl2 = document.getElementById('modal-features');
    if (fEl2 && data.topic_distribution && Object.keys(data.topic_distribution).length) {
        const topicHtml = Object.entries(data.topic_distribution)
            .sort((a, b) => b[1] - a[1])
            .map(([topic, count]) => `<span style="display:inline-block;padding:0.15rem 0.5rem;margin:0.15rem;border-radius:10px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.15);font-size:0.68rem;color:var(--text-secondary);">${topic} <strong style="color:var(--text-accent)">${count}</strong></span>`)
            .join('');
        fEl2.innerHTML += `<div style="margin-top:0.5rem;"><div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary);margin-bottom:0.3rem;font-weight:600;">Document Topics</div>${topicHtml}</div>`;
    }

    // Score reasoning + Citations from ChromaDB
    const summarySection = document.getElementById('modal-summary-section');
    const citationsEl = document.getElementById('modal-citations');

    summarySection.style.display = '';

    // Show score reasoning above citations
    const reasoningEl = document.getElementById('modal-summary-text');
    if (reasoningEl && data.score_reasoning) {
        reasoningEl.style.display = '';
        reasoningEl.textContent = data.score_reasoning;
    } else if (reasoningEl) {
        reasoningEl.style.display = 'none';
    }

    citationsEl.innerHTML = '<p style="color:var(--text-secondary);font-size:0.82rem;">Loading citations...</p>';

    // Load document citations
    fetch(`/api/person/${encodeURIComponent(data.name)}/citations`)
        .then(r => r.json())
        .then(result => {
            if (result.citations && result.citations.length) {
                const sourceLabel = { doj_pdf: 'DOJ/EFTA Document', jmail_court: 'Court Record', jmail_doj: 'DOJ/EFTA Document', court: 'Court Filing', wikipedia: 'Wikipedia', doj_press: 'DOJ Press', doj_pdf_text: 'DOJ Document' };

                // Deduplicate: same EFTA ID or URL or very similar quote start
                const seenEfta = new Set();
                const seenUrls = new Set();
                const deduplicated = result.citations.filter(c => {
                    const efta = c.efta_id || '';
                    const url = c.url || '';
                    if (efta && seenEfta.has(efta)) return false;
                    if (url && seenUrls.has(url)) return false;
                    if (efta) seenEfta.add(efta);
                    if (url) seenUrls.add(url);
                    return true;
                }).slice(0, 8);

                if (!deduplicated.length) {
                    citationsEl.innerHTML = '<p style="color:var(--text-secondary);font-size:0.82rem;">No document citations found in corpus.</p>';
                    return;
                }

                citationsEl.innerHTML = deduplicated.map(c => {
                    const label = typeof c.source === 'string' ? (sourceLabel[c.source] || c.source || 'Document') : 'Document';
                    const efta = c.efta_id || '';
                    const url = c.url || '';
                    const topic = c.topic || '';
                    const docSummary = c.doc_summary || '';
                    const quote = (c.quote || '').replace(/\\n/g, ' ').replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
                    const truncQuote = quote.length > 200 ? quote.slice(0, 200) + '...' : quote;
                    const topicBadge = topic ? `<span style="font-size:0.55rem;padding:0.1rem 0.35rem;border-radius:8px;background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.2);color:var(--accent-cyan);text-transform:uppercase;letter-spacing:0.04em;">${topic}</span>` : '';
                    return `
                        <div class="citation-item">
                            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;flex-wrap:wrap;">
                                <span class="citation-type">${label}</span>
                                ${efta ? `<span class="citation-id">${efta}</span>` : ''}
                                ${topicBadge}
                            </div>
                            ${docSummary ? `<p style="font-size:0.76rem;color:var(--text-primary);margin:0.2rem 0 0.3rem;line-height:1.5;">${docSummary}</p>` : ''}
                            ${truncQuote ? `<p class="citation-snippet">"${truncQuote}"</p>` : ''}
                            ${url ? `<a href="${url}" target="_blank" rel="noopener noreferrer" class="citation-link">View DOJ Document${efta ? ` (${efta})` : ''} &rarr;</a>` : ''}
                        </div>`;
                }).join('');
            } else {
                citationsEl.innerHTML = '<p style="color:var(--text-secondary);font-size:0.82rem;">No document citations found in corpus.</p>';
            }
        })
        .catch(() => {
            citationsEl.innerHTML = '<p style="color:var(--text-secondary);font-size:0.82rem;">Citations unavailable.</p>';
        });

    // Load consequence source links
    fetch(`/api/person/${encodeURIComponent(data.name)}/consequence-sources`)
        .then(r => r.json())
        .then(result => {
            if (result.sources && result.sources.length) {
                const sourcesHtml = result.sources.map(s =>
                    `<a href="${s.url}" target="_blank" rel="noopener noreferrer" class="citation-link" style="display:block;margin-bottom:0.3rem;">${s.title} <small style="opacity:0.6">(${s.source})</small> &rarr;</a>`
                ).join('');
                const cEl = document.getElementById('modal-consequence');
                if (cEl) {
                    cEl.innerHTML += `<div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid var(--border-subtle);">
                        <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary);margin-bottom:0.4rem;font-weight:600;">Sources</div>
                        ${sourcesHtml}
                    </div>`;
                }
            }
        })
        .catch(() => {});

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
        number: { suffix: ' / 10', font: { size: 22, color: document.documentElement.dataset.theme === 'light' ? '#0f172a' : '#f8fafc' } },
        gauge: {
            axis: { range: [0, 10], tickcolor: document.documentElement.dataset.theme === 'light' ? '#cbd5e1' : '#1e293b', dtick: 2 },
            bar: { color: LEVEL_COLORS[level] || '#475569' },
            bgcolor: document.documentElement.dataset.theme === 'light' ? '#f1f5f9' : '#0f172a', borderwidth: 0,
            steps: [
                { range: [0, 1],   color: 'rgba(71,85,105,0.15)' },
                { range: [1, 2.5], color: 'rgba(6,182,212,0.08)' },
                { range: [2.5, 5], color: 'rgba(59,130,246,0.08)' },
                { range: [5, 7.5], color: 'rgba(245,158,11,0.08)' },
                { range: [7.5, 10], color: 'rgba(239,68,68,0.08)' }
            ]
        }
    }], { ...getPlotlyTheme(), height: 165, margin: { t: 10, b: 0, l: 20, r: 20 } },
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

    // Show our 3 production models + Legal-BERT as documented failure
    const models = ['logistic_baseline', 'random_forest_tfidf', 'sentence_transformer_svc', 'legal_bert'];
    const labels = {
        logistic_baseline: 'Logistic Regression',
        random_forest_tfidf: 'Random Forest + TF-IDF',
        sentence_transformer_svc: 'ST + SVC (Semantic)',
        legal_bert: 'Legal-BERT (Fine-tuned)'
    };
    const bestModel = 'logistic_baseline'; // highest F1

    container.innerHTML = models.filter(m => data[m]).map(m => {
        const met = data[m];
        if (!met.accuracy && !met.f1_macro) return ''; // skip v2 entries without test metrics
        const acc = met.accuracy != null ? (met.accuracy * 100).toFixed(1) : '—';
        const f1 = met.f1_macro != null ? (met.f1_macro * 100).toFixed(1) : '—';
        const mcc = (met.mcc || 0).toFixed(3);
        const prec = ((met.precision_positive || 0) * 100).toFixed(1);
        const rec = ((met.recall_positive || 0) * 100).toFixed(1);
        const best = m === bestModel;
        const deprecated = m === 'legal_bert';

        return `<div class="model-card${best ? ' best' : ''}${deprecated ? ' deprecated' : ''}">
            <div class="model-card-header">
                <span class="model-card-name">${labels[m]}${best ? ' ★ Best F1' : ''}${deprecated ? ' — Deprecated' : ''}</span>
                <span class="model-card-acc" style="${deprecated ? 'color:#ef4444' : ''}">${acc}%</span>
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

    const models = ['logistic_baseline', 'random_forest_tfidf', 'sentence_transformer_svc', 'legal_bert'].filter(m => data[m] && data[m].accuracy != null);
    const labels = { logistic_baseline: 'Logistic', random_forest_tfidf: 'RF+TF-IDF', sentence_transformer_svc: 'ST+SVC', legal_bert: 'Legal-BERT' };

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

    const theme = getPlotlyTheme();
    const gridColor = document.documentElement.dataset.theme === 'light' ? '#e2e8f0' : '#1e293b';
    Plotly.newPlot('metricsBarChart', traces, {
        ...theme,
        barmode: 'group',
        xaxis: { color: theme.font.color },
        yaxis: { title: 'Score', color: theme.font.color, gridcolor: gridColor, range: [0, 1.05] },
        height: 380,
        margin: { t: 20, b: 50, l: 50, r: 20 },
        legend: { bgcolor: 'transparent', font: { size: 10, color: theme.font.color }, orientation: 'h', y: -0.2 }
    }, { responsive: true, displaylogo: false });
}

function createConfusionMatrices() {
    const data = STATE.modelResults;
    if (!data || data.error) return;
    const container = document.getElementById('confusionMatrices');
    if (!container) return;

    const models = ['logistic_baseline', 'random_forest_tfidf', 'sentence_transformer_svc', 'legal_bert'].filter(m => data[m] && data[m].confusion_matrix);
    const labels = { logistic_baseline: 'Logistic Regression', random_forest_tfidf: 'Random Forest + TF-IDF', sentence_transformer_svc: 'ST + SVC', legal_bert: 'Legal-BERT' };

    container.innerHTML = models.map(m => {
        return `<div class="confusion-panel">
            <h4>${labels[m]}</h4>
            <div id="cm-${m}" style="width:100%;height:280px;"></div>
        </div>`;
    }).join('');

    const cmTheme = getPlotlyTheme();
    const isLightCM = document.documentElement.dataset.theme === 'light';
    models.forEach(m => {
        const cm = data[m].confusion_matrix;
        const z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]];

        Plotly.newPlot(`cm-${m}`, [{
            z: z, type: 'heatmap',
            colorscale: isLightCM ? [[0, '#e0f2fe'], [0.5, '#3b82f6'], [1, '#1e40af']] : [[0, '#0f172a'], [0.5, '#1e40af'], [1, '#3b82f6']],
            showscale: false, xgap: 2, ygap: 2,
            hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        }], {
            ...cmTheme,
            xaxis: { title: 'Predicted', tickvals: [0, 1], ticktext: ['No Cons.', 'Cons.'], color: cmTheme.font.color, side: 'bottom' },
            yaxis: { title: 'Actual', tickvals: [0, 1], ticktext: ['No Cons.', 'Cons.'], color: cmTheme.font.color, autorange: 'reversed' },
            height: 260, margin: { t: 10, b: 50, l: 60, r: 10 },
            annotations: z.flatMap((row, i) => row.map((val, j) => ({
                x: j, y: i, text: String(val), showarrow: false,
                font: { color: val > 10 ? '#fff' : (isLightCM ? '#0f172a' : '#e2e8f0'), size: 16, family: 'Space Grotesk' }
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

    const ablTheme = getPlotlyTheme();
    const ablGrid = document.documentElement.dataset.theme === 'light' ? '#e2e8f0' : '#1e293b';
    Plotly.newPlot('ablationChart', [
        {
            x: labels, y: f1s, type: 'bar',
            marker: { color: colors, line: { color: ablTheme.paper_bgcolor, width: 1 } },
            text: f1s.map(v => v.toFixed(3)), textposition: 'outside',
            textfont: { color: ablTheme.font.color, size: 10 },
            hovertemplate: '<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>'
        },
        {
            x: labels, y: Array(labels.length).fill(baseF1),
            type: 'scatter', mode: 'lines',
            line: { color: 'rgba(59,130,246,0.4)', width: 1.5, dash: 'dash' },
            showlegend: false, hoverinfo: 'skip'
        }
    ], {
        ...ablTheme,
        xaxis: { color: ablTheme.font.color, tickangle: -35 },
        yaxis: { title: 'F1 Score (Macro)', color: ablTheme.font.color, gridcolor: ablGrid, range: [0, 1.05] },
        height: 380, margin: { t: 20, b: 110, l: 50, r: 20 }, showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function createExperimentChart() {
    if (!STATE.experimentResults.length) return;
    const data = STATE.experimentResults;

    const colors = data.map(d =>
        d.correlation > 0.3 ? '#10b981' : d.correlation > 0 ? '#f59e0b' : '#ef4444'
    );

    const expTheme = getPlotlyTheme();
    const expGrid = document.documentElement.dataset.theme === 'light' ? '#e2e8f0' : '#1e293b';
    Plotly.newPlot('experimentChart', [{
        x: data.map(d => d.power_tier),
        y: data.map(d => d.correlation),
        type: 'bar',
        marker: { color: colors, line: { color: expTheme.paper_bgcolor, width: 1 } },
        text: data.map(d => d.correlation.toFixed(3)),
        textposition: 'outside',
        textfont: { color: expTheme.font.color, size: 10 },
        hovertemplate: '<b>%{x}</b><br>Correlation: %{y:.3f}<br>N=%{customdata}<extra></extra>',
        customdata: data.map(d => d.n_people)
    }], {
        ...expTheme,
        xaxis: { color: expTheme.font.color },
        yaxis: { title: 'Correlation', color: expTheme.font.color, gridcolor: expGrid, range: [-0.4, 0.5] },
        height: 340, margin: { t: 20, b: 50, l: 50, r: 20 }, showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function renderLimitations() {
    const container = document.getElementById('limitationsContent');
    if (!container) return;

    const st = STATE.modelResults?.stress_test || {};

    container.innerHTML = `
        <div class="limitation-card">
            <h4>Class Imbalance (3.4:1 ratio)</h4>
            <p>51 individuals face documented consequences vs 15 without in the labeled set. A naive majority-class baseline achieves 78.6% accuracy, setting the floor any model must beat.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>SMOTE oversampling or expanding the labeled set with more positive examples from the 2,800+ document corpus.</p>
            </div>
        </div>
        <div class="limitation-card">
            <h4>Small Labeled Set (66 individuals)</h4>
            <p>Models are trained on 66 hand-labeled people; single misclassifications cause large metric swings in F1. Test set has only 3 positive examples for ST+SVC.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>Expand labeled corpus to all 1,264 registry people using the existing ChromaDB citation evidence for annotation.</p>
            </div>
        </div>
        <div class="limitation-card">
            <h4>ST+SVC Bottleneck (F1 = 0.44)</h4>
            <p>The sentence transformer + SVC model matches the majority baseline because the test set contains only 3 positive examples — not a code bug, a data limitation.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>Fine-tune the sentence encoder on legal domain text or increase training data to unlock the semantic representation's potential.</p>
            </div>
        </div>
        <div class="limitation-card">
            <h4>Inference on Unlabeled People</h4>
            <p>RF+TF-IDF uses approximated features for 1,198 of 1,264 people (exact NLP features only available for 66). Approximation uses evidence_scores fields as proxies.</p>
            <div class="improvement">
                <strong>Improvement path:</strong>
                <p>Run full NLP feature extraction (sentiment, doc diversity, subject-line detection) on all 1,264 people using the raw corpus.</p>
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
            marker: { size: 10, color: colors[i], opacity: 0.85, line: { color: document.documentElement.dataset.theme === 'light' ? '#ffffff' : '#0f172a', width: 1 } },
            text: td.map(d => d.name),
            hovertemplate: '<b>%{text}</b><br>Evidence: %{x:.1f}<br>Consequence: %{y:.0f}<extra></extra>'
        };
    });

    const scTheme = getPlotlyTheme();
    const scGrid = document.documentElement.dataset.theme === 'light' ? '#e2e8f0' : '#1e293b';
    Plotly.newPlot('scatterPlot', traces, {
        ...scTheme,
        xaxis: { title: 'Evidence Index (NLP-derived)', color: scTheme.font.color, gridcolor: scGrid, zeroline: false },
        yaxis: { title: 'Consequence Tier', color: scTheme.font.color, gridcolor: scGrid, zeroline: false,
                 tickvals: [0, 1, 2], ticktext: ['None', 'Soft', 'Hard'] },
        height: 420, margin: { t: 20, b: 60, l: 70, r: 20 },
        hovermode: 'closest',
        legend: { bgcolor: 'transparent', font: { size: 10, color: scTheme.font.color } }
    }, { responsive: true, displaylogo: false });
}

/* ============================================================
   9. GEOGRAPHIC MAP
   ============================================================ */

function createGeoMap() {
    const el = document.getElementById('geo-map');
    if (!el) return;

    fetch('/api/geo-data')
        .then(r => r.json())
        .then(data => {
            if (!data.length) return;

            const theme = getPlotlyTheme();
            const isLight = document.documentElement.dataset.theme === 'light';

            // Map country names to ISO-3 codes for Plotly choropleth
            const countryToISO = {
                'USA': 'USA', 'UK': 'GBR', 'France': 'FRA', 'Israel': 'ISR',
                'Germany': 'DEU', 'Canada': 'CAN', 'Australia': 'AUS',
                'Russia': 'RUS', 'Saudi Arabia': 'SAU', 'UAE': 'ARE',
                'Brazil': 'BRA', 'Mexico': 'MEX', 'Spain': 'ESP', 'Italy': 'ITA',
                'Japan': 'JPN', 'China': 'CHN', 'India': 'IND', 'Sweden': 'SWE',
                'Netherlands': 'NLD', 'Switzerland': 'CHE', 'Austria': 'AUT',
                'Belgium': 'BEL', 'Denmark': 'DNK', 'Norway': 'NOR', 'Finland': 'FIN',
                'Portugal': 'PRT', 'Greece': 'GRC', 'Poland': 'POL', 'Czech Republic': 'CZE',
                'Hungary': 'HUN', 'Romania': 'ROU', 'Ukraine': 'UKR', 'Turkey': 'TUR',
                'South Africa': 'ZAF', 'Nigeria': 'NGA', 'Egypt': 'EGY', 'Morocco': 'MAR',
                'Argentina': 'ARG', 'Chile': 'CHL', 'Colombia': 'COL', 'Peru': 'PER',
                'New Zealand': 'NZL', 'Singapore': 'SGP', 'South Korea': 'KOR',
                'Thailand': 'THA', 'Philippines': 'PHL', 'Vietnam': 'VNM',
                'Pakistan': 'PAK', 'Bangladesh': 'BGD', 'Sri Lanka': 'LKA',
                'Iran': 'IRN', 'Iraq': 'IRQ', 'Jordan': 'JOR', 'Lebanon': 'LBN',
                'Kuwait': 'KWT', 'Qatar': 'QAT', 'Bahrain': 'BHR', 'Oman': 'OMN',
                'Unknown': null
            };

            const locations = [], z = [], text = [], customdata = [];
            data.forEach(d => {
                const iso = countryToISO[d.country];
                if (!iso) return;
                locations.push(iso);
                z.push(d.count);
                customdata.push(d);
                const top = d.names.slice(0, 3).map(n => `  • ${n.name} (${n.impunity})`).join('<br>');
                text.push(`<b>${d.country}</b><br>Individuals: ${d.count}<br>Avg Impunity: ${d.avg_impunity}<br>No Consequence: ${d.no_consequence}<br>${top}`);
            });

            Plotly.newPlot('geo-map', [{
                type: 'choropleth',
                locations: locations,
                z: z,
                text: text,
                customdata: customdata,
                hovertemplate: '%{text}<extra></extra>',
                colorscale: [
                    [0, isLight ? '#dbeafe' : '#1e3a5f'],
                    [0.3, '#3b82f6'],
                    [0.6, '#f59e0b'],
                    [1, '#ef4444']
                ],
                colorbar: {
                    title: { text: 'Individuals', font: { color: theme.font.color, size: 11 } },
                    tickfont: { color: theme.font.color, size: 10 },
                    len: 0.6, thickness: 12,
                },
                marker: { line: { color: isLight ? '#cbd5e1' : '#334155', width: 0.5 } },
            }], {
                ...theme,
                geo: {
                    showframe: false,
                    showcoastlines: true,
                    coastlinecolor: isLight ? '#cbd5e1' : '#334155',
                    showland: true,
                    landcolor: isLight ? '#f1f5f9' : '#1e293b',
                    showocean: true,
                    oceancolor: isLight ? '#e0f2fe' : '#0f172a',
                    showcountries: true,
                    countrycolor: isLight ? '#cbd5e1' : '#334155',
                    projection: { type: 'natural earth' },
                    bgcolor: theme.paper_bgcolor,
                },
                height: 500,
                margin: { t: 10, b: 10, l: 0, r: 0 },
            }, { responsive: true, displaylogo: false });
        })
        .catch(err => console.warn('Geo map failed:', err));
}

/* ============================================================
   10. SEMANTIC SPACE (t-SNE)
   ============================================================ */

function createSemanticSpace() {
    const el = document.getElementById('semantic-plot');
    if (!el) return;

    el.innerHTML = '<p style="text-align:center;padding:4rem;color:var(--text-secondary);font-size:0.88rem;">Computing t-SNE projection... (first load may take ~30 seconds)</p>';

    fetch('/api/semantic-space')
        .then(r => r.json())
        .then(points => {
            if (!Array.isArray(points) || !points.length) return;

            const theme = getPlotlyTheme();
            const isLight = document.documentElement.dataset.theme === 'light';
            const levelColors = {
                critical: '#ef4444', high: '#f59e0b',
                moderate: '#3b82f6', low: '#06b6d4',
                minimal: isLight ? '#cbd5e1' : '#334155'
            };
            const levelOrder = ['critical', 'high', 'moderate', 'low', 'minimal'];
            const levelLabels = {
                critical: 'Critical', high: 'High',
                moderate: 'Moderate', low: 'Low', minimal: 'Minimal (background)'
            };

            const traces = levelOrder.map(level => {
                const pts = points.filter(p => p.level && p.level.toLowerCase() === level);
                if (!pts.length) return null;
                const isMinimal = level === 'minimal';
                return {
                    x: pts.map(p => p.x),
                    y: pts.map(p => p.y),
                    mode: 'markers',
                    type: 'scatter',
                    name: levelLabels[level],
                    visible: isMinimal ? 'legendonly' : true,
                    marker: {
                        size: isMinimal ? 4 : pts.map(p => Math.max(5, Math.min(18, 4 + p.impunity * 1.5))),
                        color: levelColors[level],
                        opacity: isMinimal ? 0.35 : 0.8,
                        line: { color: isLight ? '#fff' : '#0f172a', width: isMinimal ? 0 : 0.5 }
                    },
                    text: pts.map(p => p.name),
                    customdata: pts.map(p => p.impunity),
                    hovertemplate: '<b>%{text}</b><br>Impunity: %{customdata:.1f}<extra></extra>'
                };
            }).filter(Boolean);

            Plotly.newPlot('semantic-plot', traces, {
                ...theme,
                xaxis: { visible: false, zeroline: false },
                yaxis: { visible: false, zeroline: false },
                height: 550,
                margin: { t: 20, b: 20, l: 20, r: 20 },
                hovermode: 'closest',
                legend: { bgcolor: 'transparent', font: { size: 11, color: theme.font.color } },
                annotations: [{
                    text: 'Point size = impunity index · Proximity = similar document contexts',
                    showarrow: false, x: 0.5, y: -0.02, xref: 'paper', yref: 'paper',
                    font: { size: 10, color: theme.font.color }, xanchor: 'center'
                }]
            }, { responsive: true, displaylogo: false });

            // Click to open person modal
            el.on('plotly_click', evt => {
                const pt = evt.points[0];
                if (pt && pt.text) openPersonModal(pt.text);
            });
        })
        .catch(err => {
            el.innerHTML = '<p style="text-align:center;padding:2rem;color:var(--text-secondary);">Semantic space unavailable.</p>';
            console.warn('Semantic space failed:', err);
        });
}

/* ============================================================
   11. SIDE NAVIGATION
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

function setupThemeToggle() {
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.dataset.theme = saved;

    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', () => {
        const current = document.documentElement.dataset.theme;
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.dataset.theme = next;
        localStorage.setItem('theme', next);

        // Re-render charts with new theme colors
        if (STATE.people.length) {
            initNetworkGraph();
            createMetricsBarChart();
            createConfusionMatrices();
            createAblationChart();
            createExperimentChart();
            createScatterPlot();
        }
    });
}

/* ============================================================
   CONTENT WARNING GATE
   ============================================================ */

function setupContentGate() {
    const gate = document.getElementById('content-gate');
    if (!gate) return;

    // Check if user already acknowledged in this session
    if (sessionStorage.getItem('content-gate-acknowledged')) {
        gate.classList.add('dismissed');
        return;
    }

    const checkbox = document.getElementById('gate-age-check');
    const btn = document.getElementById('gate-continue');
    if (!checkbox || !btn) return;

    checkbox.addEventListener('change', () => {
        btn.disabled = !checkbox.checked;
    });

    btn.addEventListener('click', () => {
        sessionStorage.setItem('content-gate-acknowledged', 'true');
        gate.classList.add('dismissed');
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setupContentGate();
    setupThemeToggle();
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
