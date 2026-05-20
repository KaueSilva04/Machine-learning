// -------------------------------------------------------------
// FRONTEND APP CONTROLLER - GA + IA QR CODE OPTIMIZER
// -------------------------------------------------------------

let evolutionChart = null;
let currentSandboxFile = null;
let debounceTimeout = null;

const initApp = () => {
    // 1. Inicializar Gráfico de Evolução vazio
    inicializarGrafico();

    // 2. Carregar estatísticas e comparativos iniciais do Dashboard
    carregarEstatisticas();
    carregarComparativo();

    // 3. Configurar eventos do Algoritmo Genético
    const btn = document.getElementById("btn-start-ga");
    if (btn) {
        btn.addEventListener("click", iniciarAlgoritmoGenetico);
    }

    // 4. Configurar Drag and Drop da Sandbox de Filtros
    configurarSandboxUpload();
};

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
} else {
    initApp();
}

// --- INICIALIZAÇÃO DO GRÁFICO (CHART.JS) ---
function inicializarGrafico() {
    const ctx = document.getElementById("evolutionChart").getContext("2d");
    
    // Gradients para as linhas do gráfico
    const gradientViolet = ctx.createLinearGradient(0, 0, 0, 300);
    gradientViolet.addColorStop(0, "rgba(139, 92, 246, 0.4)");
    gradientViolet.addColorStop(1, "rgba(139, 92, 246, 0.0)");

    const gradientCyan = ctx.createLinearGradient(0, 0, 0, 300);
    gradientCyan.addColorStop(0, "rgba(6, 182, 212, 0.4)");
    gradientCyan.addColorStop(1, "rgba(6, 182, 212, 0.0)");

    evolutionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Melhor Evolução (Com IA)',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: gradientViolet,
                    fill: true,
                    tension: 0.35,
                    borderWidth: 3,
                    pointBackgroundColor: '#8b5cf6',
                    pointHoverRadius: 7
                },
                {
                    label: 'Melhor Evolução (Sem IA)',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: gradientCyan,
                    fill: true,
                    tension: 0.35,
                    borderWidth: 3,
                    pointBackgroundColor: '#06b6d4',
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Outfit', size: 12 }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } },
                    title: { display: true, text: 'Geração', color: '#64748b' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } },
                    title: { display: true, text: 'Score de Aptidão (0 a 1.0)', color: '#64748b' },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

// --- CARREGAMENTO DE DADOS ---
async function carregarEstatisticas() {
    try {
        const response = await fetch("/api/stats");
        const stats = await response.json();
        
        if (stats.error) return;

        document.getElementById("val-total-extractions").innerText = Number(stats.total_extractions).toLocaleString();
        document.getElementById("val-taxa-sucesso").innerText = `${stats.taxa_sucesso_geral}%`;
        document.getElementById("val-total-imagens").innerText = stats.total_imagens;
        
        // Exibe badge de CUDA e WeChat baseado nos dados reais
        const cudaBadge = document.getElementById("cuda-badge");
        if (stats.total_imagens > 0) {
            cudaBadge.style.display = "inline-flex";
        }
    } catch (e) {
        console.error("Erro ao carregar estatísticas", e);
    }
}

async function carregarComparativo() {
    try {
        const response = await fetch("/api/comparativo");
        const comp = await response.json();

        if (comp.error) return;

        const ganhoScore = comp.ganho_melhor_score * 100;
        const ganhoTempo = comp.ganho_tempo_porcento;

        const valGanhoIa = document.getElementById("val-ganho-ia");
        const valTempoEconomizado = document.getElementById("val-tempo-economizado");

        if (comp.com_nn.total > 0 && comp.sem_nn.total > 0) {
            valGanhoIa.innerText = `${ganhoScore > 0 ? '+' : ''}${ganhoScore.toFixed(1)}% no Score`;
            valTempoEconomizado.innerText = `${ganhoTempo.toFixed(1)}% mais rápido com IA`;
            valTempoEconomizado.style.color = "var(--color-cyan)";
        } else {
            valGanhoIa.innerText = "Aguardando AG";
            valTempoEconomizado.innerText = "Treine COM e SEM rede para comparar";
            valTempoEconomizado.style.color = "var(--text-muted)";
        }

        // --- ATUALIZA O GRÁFICO COMPARATIVO COM AS DUAS LINHAS HISTÓRICAS ---
        const histCom = comp.com_nn.historico || [];
        const histSem = comp.sem_nn.historico || [];

        const maxGen = Math.max(histCom.length, histSem.length, 1);
        const labels = [];
        for (let i = 1; i <= maxGen; i++) {
            labels.push(i);
        }

        evolutionChart.data.labels = labels;
        evolutionChart.data.datasets[0].data = labels.map((_, idx) => histCom[idx] ? histCom[idx].melhor_score : null);
        evolutionChart.data.datasets[1].data = labels.map((_, idx) => histSem[idx] ? histSem[idx].melhor_score : null);
        evolutionChart.update();

    } catch (e) {
        console.error("Erro ao carregar comparativo", e);
    }
}

// --- EXECUÇÃO DO ALGORITMO GENÉTICO (SSE) ---
function iniciarAlgoritmoGenetico() {
    const btn = document.getElementById("btn-start-ga");
    const pop = document.getElementById("ga-pop").value;
    const gen = document.getElementById("ga-gen").value;
    const useNn = document.getElementById("ga-use-nn").checked;

    // Bloquear controles durante execução
    btn.disabled = true;
    btn.classList.add("disabled");
    document.getElementById("ga-pop").disabled = true;
    document.getElementById("ga-gen").disabled = true;
    document.getElementById("ga-use-nn").disabled = true;

    // Reset de status e console
    const statusContainer = document.getElementById("ga-status-container");
    statusContainer.style.display = "flex";
    const consoleOutput = document.getElementById("ga-console-output");
    consoleOutput.innerText = "🧬 Inicializando Algoritmo Genético...\n";

    const progressFill = document.getElementById("ga-progress-fill");
    const progressPercent = document.getElementById("ga-progress-percent");
    const genLabel = document.getElementById("ga-generation-label");

    progressFill.style.width = "0%";
    progressPercent.innerText = "0%";
    genLabel.innerText = `Geração: 0 / ${gen}`;

    // Configura os labels do gráfico para o tamanho do experimento atual
    const chartLabels = [];
    for (let i = 1; i <= gen; i++) {
        chartLabels.push(i);
    }
    evolutionChart.data.labels = chartLabels;

    // Limpa apenas a linha ativa do experimento atual, mantendo a outra como baseline comparativo!
    if (useNn) {
        evolutionChart.data.datasets[0].data = [];
    } else {
        evolutionChart.data.datasets[1].data = [];
    }
    evolutionChart.update();

    // Conectar canal Server-Sent Events (SSE)
    const eventSource = new EventSource(`/api/run-ga?use_nn=${useNn}&pop=${pop}&gen=${gen}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
            consoleOutput.innerText += `\n❌ ERRO: ${data.error}\n`;
            eventSource.close();
            liberarControlesGA();
            return;
        }

        if (data.done) {
            const res = data.result;
            consoleOutput.innerText += `\n🏆 PROCESSO CONCLUÍDO!\n`;
            consoleOutput.innerText += `🔹 Tempo Total: ${res.tempo_total}s\n`;
            consoleOutput.innerText += `🔹 Melhor Score Final: ${(res.melhor_score_final * 100).toFixed(1)}%\n`;
            consoleOutput.innerText += `🔹 Média de Score Final: ${(res.media_score_final * 100).toFixed(1)}%\n\n`;
            consoleOutput.innerText += `🔹 Configuração Otimizada Vencedora:\n`;
            
            for (const [key, val] of Object.entries(res.vencedor)) {
                consoleOutput.innerText += `   🔸 ${key}: ${val}\n`;
            }

            consoleOutput.scrollTop = consoleOutput.scrollHeight;
            eventSource.close();
            liberarControlesGA();
            
            // Recarrega dados e KPI do Dashboard instantaneamente
            carregarEstatisticas();
            carregarComparativo();
            return;
        }

        // Progresso de uma Geração do AG
        if (data.generation) {
            const g = data.generation;
            const totalG = data.total_generations;
            const pct = Math.round((g / totalG) * 100);

            // Atualiza barra de progresso
            progressFill.style.width = `${pct}%`;
            progressPercent.innerText = `${pct}%`;
            genLabel.innerText = `Geração: ${g} / ${totalG}`;

            // Imprime no console
            consoleOutput.innerText += `\n🧬 Geração ${String(g).padStart(2, '0')}/${totalG} concluída!\n`;
            consoleOutput.innerText += `   🏆 Melhor Score: ${(data.best_score * 100).toFixed(1)}% | 📈 Média: ${(data.avg_score * 100).toFixed(1)}%\n`;
            consoleOutput.innerText += `   💡 Melhor Filtro: ${JSON.stringify(data.best_config)}\n`;
            consoleOutput.scrollTop = consoleOutput.scrollHeight;

            // Corrida de otimização em tempo real no gráfico!
            if (useNn) {
                evolutionChart.data.datasets[0].data[g - 1] = data.best_score;
            } else {
                evolutionChart.data.datasets[1].data[g - 1] = data.best_score;
            }
            evolutionChart.update();
        }
    };

    eventSource.onerror = (e) => {
        consoleOutput.innerText += `\n⚠️ Desconexão inesperada do stream ou processo concluído.\n`;
        eventSource.close();
        liberarControlesGA();
    };
}

function liberarControlesGA() {
    const btn = document.getElementById("btn-start-ga");
    btn.disabled = false;
    btn.classList.remove("disabled");
    document.getElementById("ga-pop").disabled = false;
    document.getElementById("ga-gen").disabled = false;
    document.getElementById("ga-use-nn").disabled = false;
}

// --- SANDBOX DE FILTROS INTERATIVOS ---
function configurarSandboxUpload() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("sandbox-file");

    // Click abre selecionador
    dropZone.addEventListener("click", () => fileInput.click());

    // Eventos drag and drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            carregarImagemSandbox(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            carregarImagemSandbox(e.target.files[0]);
        }
    });

    // Configurar listeners de mudança nos sliders
    const sliders = [
        "slide-ksize", "slide-contrast", "slide-bright", "slide-sat",
        "slide-sharp", "slide-clahe", "slide-tblock", "slide-tc"
    ];

    sliders.forEach(id => {
        const input = document.getElementById(id);
        const valSpan = document.getElementById(id.replace("slide-", "val-"));

        input.addEventListener("input", (e) => {
            valSpan.innerText = e.target.value;
            
            // Debounce para evitar sobrecarga no servidor FastAPI durante deslizamento contínuo
            clearTimeout(debounceTimeout);
            debounceTimeout = setTimeout(atualizarFiltrosSandbox, 80);
        });
    });
}

function carregarImagemSandbox(file) {
    currentSandboxFile = file;

    // Renderizar preview da imagem original
    const reader = new FileReader();
    reader.onload = (e) => {
        const rawBox = document.getElementById("img-box-raw");
        rawBox.innerHTML = `<img src="${e.target.result}" alt="Original">`;
    };
    reader.readAsDataURL(file);

    // Habilitar a área de sliders
    document.getElementById("sliders-box").classList.remove("disabled");
    
    // Dispara primeiro processamento automático com os sliders default
    atualizarFiltrosSandbox();
}

async function atualizarFiltrosSandbox() {
    if (!currentSandboxFile) return;

    const processedBox = document.getElementById("img-box-processed");
    processedBox.innerHTML = `<div class="img-placeholder">Processando via CUDA...</div>`;

    const formData = new FormData();
    formData.append("file", currentSandboxFile);
    formData.append("kernel_size", document.getElementById("slide-ksize").value);
    formData.append("contrast", document.getElementById("slide-contrast").value);
    formData.append("bright", document.getElementById("slide-bright").value);
    formData.append("sat", document.getElementById("slide-sat").value);
    formData.append("sharp", document.getElementById("slide-sharp").value);
    formData.append("clahe", document.getElementById("slide-clahe").value);
    formData.append("thresh_block", document.getElementById("slide-tblock").value);
    formData.append("thresh_c", document.getElementById("slide-tc").value);

    try {
        const response = await fetch("/api/test-filter", {
            method: "POST",
            body: formData
        });
        const res = await response.json();

        if (res.success) {
            // Renderiza imagem binarizada processada
            processedBox.innerHTML = `<img src="${res.image}" alt="Processed">`;

            // Atualiza caixa de status de decodificação
            const decodeBox = document.getElementById("decode-result-box");
            const statusIcon = document.getElementById("decode-status-icon");
            const statusText = document.getElementById("decode-status-text");

            decodeBox.style.display = "flex";

            if (res.has_decoded) {
                decodeBox.className = "decode-status-box success";
                statusIcon.innerText = "✅";
                statusText.innerText = res.decoded_text;
            } else {
                decodeBox.className = "decode-status-box failure";
                statusIcon.innerText = "❌";
                statusText.innerText = "Não foi possível decodificar o QR Code com estes filtros.";
            }
        } else {
            processedBox.innerHTML = `<div class="img-placeholder text-rose-500">${res.message}</div>`;
        }
    } catch (e) {
        console.error("Erro ao rodar sandbox de filtros", e);
        processedBox.innerHTML = `<div class="img-placeholder">Erro de conexão com o servidor.</div>`;
    }
}
