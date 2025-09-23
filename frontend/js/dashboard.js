    document.getElementById("saludo").textContent = "Welcome!!";

    function logout() {
        sessionStorage.clear();
        window.location.href = 'index.html';
    }

    const apiBase = 'http://localhost:8080/api';

    async function fetchData() {
        try {
            const res = await fetch(`${apiBase}/detections`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener los datos", err);
            return [];
        }
    }

    async function fetchTotals() {
        try {
            const res = await fetch(`${apiBase}/totales`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener totales:", err);
            return [];
        }
    }

    async function fetchProportions() {
        try {
            const res = await fetch(`${apiBase}/proporciones`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener proporciones:", err);
            return [];
        }
    }

    async function fetchByDate() {
        try {
            const res = await fetch(`${apiBase}/por-fecha`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener datos por fecha:", err);
            return [];
        }
    }

    async function fetchByDayWeek() {
        try {
            const res = await fetch(`${apiBase}/por-dia-semana`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener los datos por dia", err);
            return [];
        }
    }

    async function fetchByWeek() {
        try {
            const res = await fetch(`${apiBase}/conteo/entre-semana`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener los datos entre semana", err);
            return [];
        }
    }

    async function fetchByWeekend() {
        try {
            const res = await fetch(`${apiBase}/conteo/fin-semana`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener los datos de fin de semana", err);
            return [];
        }
    }

    async function fetchByDayEtiqueta() {
        try {
            const res = await fetch(`${apiBase}/conteo/desecho-dia`);
            return await res.json();
        } catch (err) {
            console.error("Error al obtener datos día-etiqueta:", err);
            return [];
        }
    }

    function generarBarChart(labels, valores) {
        new Chart(document.getElementById("barChart"), {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: "Conteos por Etiqueta",
                    data: valores
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Detecciones Totales por Etiqueta'
                    }
                },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    function generarPieChart(labels, valores) {
        new Chart(document.getElementById("pieChart"), {
            type: "pie",
            data: {
                labels,
                datasets: [{ data: valores }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Proporción de Detecciones'
                    }
                }
            }
        });
    }

    function generarLineChart(labels, valores) {
        new Chart(document.getElementById("lineChart"), {
            type: "line",
            data: {
                labels,
                datasets: [{
                    label: "Conteos por Día",
                    data: valores,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Evolución Temporal de Detecciones'
                    }
                }
            }
        });
    }

    function generarRadarChart(labels, valores) {
    new Chart(document.getElementById("radarChart"), {
        type: "radar",
        data: {
            labels,
            datasets: [{
                label: "Detecciones por Día de Semana",
                data: valores,
                fill: true,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Detección Semanal'
                }
            }
        }
    });
}

    function generarDoughnutChart(entre, fin) {
        new Chart(document.getElementById("doughnutChart"), {
        type: "doughnut",
        data: {
            labels: ["Entre semana", "Fin de semana"],
            datasets: [{
                data: [entre.conteo, fin.conteo],
                backgroundColor: ["#36A2EB", "#FF6384"]
            }]
        },
        options: {
            plugins: {
                title: {
                    display: true,
                    text: 'Comparación: Entre Semana vs Fin de Semana'
                }
            }
        }
        });
    }

    /** Crea el layout del Grouped Chart */
    function generarGroupedChart(data) {
        const dias = [...new Set(data.map(d => d.dia))];
        const etiquetas = [...new Set(data.map(d => d.etiqueta))];
    const datasets = etiquetas.map(etiqueta => {
        return {
            label: etiqueta,
            data: dias.map(dia =>
                data.find(d => d.dia === dia && d.etiqueta === etiqueta)?.conteo || 0
            )
        };
    });
    new Chart(document.getElementById("groupedChart"), {
        type: "bar",
        data: {
            labels: dias,
            datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Basura Detectada por Día y Etiqueta'
                }
            }
        }
    });
}

    /**
     * Construye una tabla HTML con cabeceras y filas a partir de un array de objetos.
     * @param {Object[]} data  — Array de objetos con mismas llaves.
     * @param {string} idTabla — ID de la tabla (<table>) en el DOM.
     */
    function crearTablaDynamic(data, idTabla) {
        const table = document.getElementById(idTabla);
        if (!data.length) {
            table.innerHTML = '<p>No hay datos para mostrar.</p>';
            return;
        }

        // Limpia los datos anteriores
        table.innerHTML = '';

        // 1) Crea THEAD con las keys
        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        const keys = Object.keys(data[0]);
        for (const key of keys) {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        }

        // 2) Crea TBODY y filas de los datos
        const tbody = table.createTBody();
        data.forEach(item => {
            const row = tbody.insertRow();
            for (const key of keys) {
                const cell = row.insertCell();
                cell.textContent = item[key];
            }
        });
    }

    /** Carga y renderiza los charts */
    window.addEventListener('DOMContentLoaded', async () => {
        /** Constantes de datos */
        const diaSemana = await fetchByDayWeek();
        const byDate = await fetchByDate();
        const fechas = byDate.map(d => d.fecha);
        const conteos = byDate.map(d => d.conteo);
        const proporciones = await fetchProportions();
        const totales = await fetchTotals();
        const datosDiaEtiqueta = await fetchByDayEtiqueta();
        const datosE = await fetchData();
        const entre = await fetchByWeek();
        const fin = await fetchByWeekend();


        /** Generadores de Charts */
        generarBarChart(
            totales.map(d => d.etiqueta),
            totales.map(d => d.total)
        );

        generarPieChart(
            proporciones.map(d => d.etiqueta),
            proporciones.map(d => d.proporcion * 100)
        );

        generarLineChart(
            fechas, 
            conteos
        );

        generarRadarChart(
            diaSemana.map(d => d.dia),
            diaSemana.map(d => d.conteo)
        );

        generarGroupedChart(
            datosDiaEtiqueta
        );

        generarDoughnutChart(
            entre,
            fin
        )

        /** Tabla */
        crearTablaDynamic(
            datosE, 
            "idTabla"
        );

    });