"use strict";

const data = window.FLOOD_BRIEF;
const byId = (id) => document.getElementById(id);
const format = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const area = (value) => format.format(value);
const signed = (value) => (value > 0 ? "+" : "") + area(value);
const percent = (part, total) => area(100 * part / total) + "%";
const [mapWidth, mapHeight] = data.extent;
const info = data.caseInfo;
const buildingSources = Object.keys(data.exposure?.buildings || {});
const waterLegend = [
  ["稳定水面", data.colors.stable], ["新增水面", data.colors.gain], ["减少水面", data.colors.loss],
];
const boundaryLegend = [
  ["未共同观测", data.colors.missing], ["案例区界", "#182e3d", "line"],
  ["乡镇界", "#7b7363", "dash"],
];
const views = {
  change: {
    title: "两期水面变化图",
    subtitle: info.place + " · " + info.sensor,
    alt: "SAR 水面变化图，显示乡镇边界以及各类变化水面",
    legend: [...waterLegend, ...boundaryLegend, ["居民用地轮廓", "#a48b59", "line"],
      ["参考水系／岸线", "#49758b", "dash"]],
    caption: "水面变化统计限于两期共同有效观测区。稳定水面为两期均识别为水面的区域。",
  },
  coverage: {
    title: "两期 SAR 覆盖快视",
    subtitle: info.sensor + " · 逐景观测与同日拼接",
    alt: "两期 SAR 影像覆盖快视图，右侧为同日观测合并",
    legend: [["案例区界", "#ce343c", "line"], ["源影像外接框", "#8daec9", "dash"],
      ["未显示有效影像", "#e5ebf0"]],
    caption: "右列为同日观测拼接。两期有效观测取交集后统计水面变化。",
  },
  exposure: {
    title: "道路与建筑潜在暴露",
    subtitle: "新增水面与设施底图叠加",
    alt: "道路与建筑潜在暴露总览图，叠加参考地形与新增水面",
    legend: [["道路相交段", "#aa2835", "line"], ["桥隧标记路段", "#6b1823", "dash"],
      ...(buildingSources.length ? [[buildingSources[0] + " 建筑重叠 ≥50%", "#ae144f"],
        [buildingSources[0] + " 建筑重叠 <50%", "#d36883"]] : []),
      ...(buildingSources.length > 1 ? [[buildingSources[1] + " 建筑相交", "#116964"]] : []), ...waterLegend, ...boundaryLegend],
    caption: "道路显示与新增水面相交的线段，建筑保留原轮廓。底图中的同一建筑可能同时出现在两套数据中。",
  },
  s1: {
    title: "主水体 · S1 辅助判读",
    subtitle: data.auxDates || "Sentinel-1",
    alt: "Sentinel-1 识别的主水体两期水面变化",
    legend: [...waterLegend, ["人工复核修正位置", "#ae3844", "dash"],
      ["参考水系／岸线", "#49758b", "dash"]],
    caption: "主水体面积按复核后结果统计。辅助观测与主数据分别按各自日期和范围解读。",
  },
};

const enabled = document.querySelectorAll('[role="tab"]');
for (const tab of enabled) {
  if (!data.figures.includes(tab.dataset.view) || (tab.dataset.view === "s1" && !data.s1) ||
      (tab.dataset.view === "exposure" && !data.exposure)) tab.remove();
}
const tabs = [...document.querySelectorAll('[role="tab"]')];
const select = byId("town-select");
const mapImage = byId("map-img");
const dialog = byId("map-dialog");
const svg = byId("town-map");
const drawing = byId("map-drawing");
const townPaths = new Map();
let view = "change";
let queryMode = false;
let townName = "";
let rankingKey = "gain";
let zoom = 1;
let mapCenter = [mapWidth / 2, mapHeight / 2];

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function svgElement(tag, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function renderLegend(items) {
  byId("legend").replaceChildren(...items.map(([label, color, style]) => {
    const li = element("li");
    const swatch = element("span", undefined, style ? "swatch " + style : "swatch");
    swatch.style.setProperty("--swatch", color);
    swatch.setAttribute("aria-hidden", "true");
    li.append(swatch, label);
    return li;
  }));
}

function renderMode() {
  const interactive = view === "change" && queryMode;
  byId("interactive").hidden = !interactive;
  byId("map-art").hidden = interactive;
  byId("map-mode").hidden = view !== "change";
  byId("map-mode").setAttribute("aria-pressed", String(interactive));
  byId("map-mode").textContent = interactive ? "回到专题图" : "分区查询";
  byId("open-map").hidden = interactive;
  if (interactive) {
    initMap();
    updateMapFrame();
    renderLegend([...waterLegend, ["共同有效观测区", data.colors.common], ...boundaryLegend]);
    byId("map-caption").textContent = "点击乡镇查看面积统计。此图为简化矢量示意，标注和地形可在专题图查看。";
  } else {
    renderLegend(views[view].legend);
    byId("map-caption").textContent = views[view].caption;
  }
}

function renderSide() {
  if (view === "change") return;
  const exposure = data.exposure || {};
  const s1 = data.s1 || {};
  const summaries = {
    coverage: {
      title: "观测范围",
      rows: [[info.dates[0], (data.scenes[0] || "—") + " 景"], [info.dates[1], (data.scenes[1] || "—") + " 景"],
        ["共同有效观测区", area(data.totals.common) + " km²"],
        ["共同有效覆盖率", percent(data.totals.common, data.totals.roi)]],
      note: "未共同观测的区域按缺测处理。逐镇查询可核对各乡镇的有效观测范围。",
    },
    exposure: {
      title: "潜在暴露统计",
      rows: [["道路与步道相交长度", area(exposure.road_overlap_km) + " km"],
        ["相交道路与步道", exposure.road_features + " 个要素"],
        ["其中带桥隧标记", exposure.bridge_tunnel_features + " 个要素"],
        ...buildingSources.flatMap((source, i) => [
          [source + " 建筑轮廓", exposure.buildings[source].features + " 个"],
          ...(i === 0 ? [["其中重叠比例 ≥50%", exposure.buildings[source].at_least_half + " 个"]] : []),
        ])],
      note: "相交结果用于确定核查对象。两套建筑底图分别统计。",
    },
    s1: {
      title: "主库区水面统计",
      rows: [["前期水面", area(s1.pre) + " km²"], ["后期水面", area(s1.post) + " km²"],
        ["水面净增", signed(s1.net) + " km²"]],
      note: "复核时修正了库心一处约 " + (data.s1Patch ?? 0).toFixed(3) + " km² 的疑似误判减少水面。",
    },
  };
  const summary = summaries[view];
  const heading = element("div", undefined, "side-head");
  heading.append(element("h2", summary.title));
  const list = element("dl", undefined, "detail-list");
  for (const [label, value] of summary.rows) {
    const row = element("div");
    row.append(element("dt", label), element("dd", value));
    list.append(row);
  }
  const other = byId("other-summary");
  other.replaceChildren(heading, list, element("p", summary.note, "detail-note"));
  if (view === "exposure" && data.figures.includes("optical")) {
    const figure = element("figure", undefined, "inset");
    const button = element("button");
    button.type = "button";
    button.setAttribute("aria-label", "查看道路和建筑潜在暴露的光学底图大图");
    const image = element("img");
    image.src = "assets/optical.webp";
    image.alt = "Sentinel-2 真彩色影像上的新增水面与道路建筑局部叠加图";
    button.append(image);
    button.addEventListener("click", () => showImage(image, "道路与建筑潜在暴露 · 局部"));
    figure.append(button, element("figcaption", info.opticalLabel));
    other.append(figure);
  }
}

function switchView(next) {
  view = next;
  for (const tab of tabs) {
    const active = tab.dataset.view === view;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  byId("map-panel").setAttribute("aria-labelledby", "tab-" + view);
  byId("map-title").textContent = views[view].title;
  byId("map-subtitle").textContent = views[view].subtitle;
  byId("stage").dataset.view = view;
  mapImage.src = "assets/" + view + ".webp";
  mapImage.alt = views[view].alt;
  byId("change-summary").hidden = view !== "change";
  byId("other-summary").hidden = view === "change";
  renderSide();
  renderMode();
}

function initMap() {
  if (townPaths.size) return;
  const frame = document.createDocumentFragment();
  const drawPath = (path, fill) => svgElement("path", { d: path, fill, "fill-rule": "evenodd" });
  frame.append(drawPath(data.roi, data.colors.missing));
  for (const key of ["common", "stable", "gain", "loss"]) {
    frame.append(drawPath(data.layers[key], data.colors[key]));
  }
  for (const town of data.towns) {
    const path = svgElement("path", { d: town.path, class: "town-hit", "fill-rule": "evenodd" });
    const title = svgElement("title", {});
    title.textContent = town.name;
    path.append(title);
    path.addEventListener("click", () => chooseTown(town.name));
    townPaths.set(town.name, path);
    frame.append(path);
  }
  for (const name of info.priority) {
    const town = data.towns.find((item) => item.name === name);
    if (!town) continue;
    const label = svgElement("text", { x: town.label[0], y: town.label[1] });
    label.textContent = town.name;
    frame.append(label);
  }
  const outline = svgElement("path", { d: data.roi, fill: "none", stroke: "#304c59",
    "stroke-width": "1.3", "vector-effect": "non-scaling-stroke", "pointer-events": "none" });
  drawing.append(frame, outline);
}

function updateMapFrame() {
  const w = mapWidth * 1.045 / zoom;
  const h = mapHeight * 1.075 / zoom;
  svg.setAttribute("viewBox", [mapCenter[0] - w / 2, mapCenter[1] - h / 2, w, h].join(" "));
  for (const [name, path] of townPaths) path.classList.toggle("is-current", name === townName);
}

function chooseTown(name) {
  const town = data.towns.find((item) => item.name === name);
  const record = town || data.totals;
  townName = name;
  select.value = name;
  byId("local-pre").textContent = record.common > 0 ? area(record.pre) : "—";
  byId("local-post").textContent = record.common > 0 ? area(record.post) : "—";
  const note = record.common > 0
    ? "共同有效覆盖 " + percent(record.common, record.roi) + (town ?
      " · 新增 " + area(record.gain) + " / 减少 " + area(record.loss) + " km²" : " · 统计范围为" + info.place)
    : "两期无共同有效观测，无法比较该乡镇水面变化。";
  byId("local-note").textContent = note;
  queryMode = true;
  mapCenter = town ? town.label : [mapWidth / 2, mapHeight / 2];
  if (!town) zoom = 1;
  renderMode();
  for (const row of document.querySelectorAll(".rank-row")) {
    row.setAttribute("aria-current", String(row.dataset.name === townName));
  }
}

function renderRanking() {
  const towns = [...data.towns]
    .filter((town) => town.common > 0 && town[rankingKey] > 0)
    .sort((a, b) => b[rankingKey] - a[rankingKey]).slice(0, 5);
  byId("rank-title").textContent = rankingKey === "gain" ? "新增水面最多的乡镇" : "减少水面最多的乡镇";
  const max = towns[0]?.[rankingKey] || 1;
  byId("ranking").replaceChildren(...towns.map((town) => {
    const li = element("li");
    const button = element("button", undefined, "rank-row");
    const track = element("span", undefined, "track");
    const bar = element("span", undefined, "bar");
    bar.style.width = (100 * town[rankingKey] / max) + "%";
    bar.style.background = data.colors[rankingKey];
    track.append(bar);
    track.setAttribute("aria-hidden", "true");
    button.type = "button";
    button.dataset.name = town.name;
    button.setAttribute("aria-current", String(town.name === townName));
    button.setAttribute("aria-label", town.name + "，" + area(town[rankingKey]) + " 平方公里");
    button.append(element("span", town.name), track, element("span", area(town[rankingKey]), "rank-value"));
    button.addEventListener("click", () => chooseTown(town.name));
    li.append(button);
    return li;
  }));
}

function showImage(image, title) {
  byId("dialog-img").src = image.src;
  byId("dialog-img").alt = image.alt;
  byId("dialog-title").textContent = title;
  dialog.showModal();
}

for (const [id, value] of Object.entries({
  "lead-net": signed(data.totals.net), "metric-net": signed(data.totals.net),
  "metric-gain": area(data.totals.gain), "metric-loss": area(data.totals.loss),
  "metric-common": area(data.totals.common),
  "metric-coverage": "占案例区分析范围 " + percent(data.totals.common, data.totals.roi),
})) byId(id).textContent = value;

for (const town of [...data.towns].sort((a, b) => a.name.localeCompare(b.name, "zh-CN"))) {
  const option = element("option", town.name);
  option.value = town.name;
  select.append(option);
}
select.addEventListener("change", () => chooseTown(select.value));

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
  tab.addEventListener("keydown", (event) => {
    const next = {
      ArrowRight: (index + 1) % tabs.length,
      ArrowLeft: (index + tabs.length - 1) % tabs.length,
      Home: 0, End: tabs.length - 1,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    tabs[next].focus();
    switchView(tabs[next].dataset.view);
  });
}

for (const button of document.querySelectorAll("[data-rank]")) {
  button.addEventListener("click", () => {
    rankingKey = button.dataset.rank;
    for (const item of document.querySelectorAll("[data-rank]")) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    renderRanking();
  });
}
byId("map-mode").addEventListener("click", () => {
  queryMode = !queryMode;
  renderMode();
});
byId("zoom-in").addEventListener("click", () => {
  zoom = Math.min(3, zoom * 1.25);
  updateMapFrame();
});
byId("zoom-out").addEventListener("click", () => {
  zoom = Math.max(1, zoom / 1.25);
  updateMapFrame();
});
byId("reset-map").addEventListener("click", () => chooseTown(""));
byId("open-map").addEventListener("click", () => showImage(mapImage, views[view].title));
byId("print").addEventListener("click", () => window.print());
dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
document.title = info.title;
byId("title").textContent = info.title;
byId("case-place").textContent = info.place;
byId("case-sensor").textContent = info.sensor;
byId("date-pre").textContent = info.dates[0];
byId("date-post").textContent = info.dates[1];
byId("case-sources").textContent = info.sources;
if (data.briefNote) byId("method-note").textContent = data.briefNote;
switchView("change");
renderRanking();
