//! HTML template for the interactive function-level call graph explorer.
//!
//! Generates a standalone HTML page with two views:
//! - **Overview**: D3.js force-directed graph of all functions
//! - **Focus**: 3-column layout (callers ← selected → callees)
//!
//! Data uses compact keys: `n`=name, `f`=file, `k`=kind, `s`=sig, `l`=lines,
//! `g`=group. Links are `[src, tgt, "c"|"t"]` (calls/types).

/// Generate the complete HTML string from pre-built JSON data.
pub(crate) fn render(nodes_json: &str, links_json: &str, groups_json: &str) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Call Graph Explorer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f0f1a;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif}}
.topbar{{display:flex;align-items:center;gap:12px;padding:8px 16px;
  background:#16162a;border-bottom:1px solid #2a2a4a}}
.topbar h1{{font-size:14px;font-weight:600;color:#7c7cff}}
.search-box{{flex:1;max-width:400px}}
.search-box input{{width:100%;padding:6px 10px;background:#1e1e3a;border:1px solid #3a3a5a;
  border-radius:4px;color:#e0e0e0;font-size:12px;outline:none}}
.search-box input:focus{{border-color:#7c7cff}}
.btn{{padding:4px 12px;background:#2a2a4a;border:1px solid #3a3a5a;
  border-radius:4px;color:#aaa;cursor:pointer;font-size:12px}}
.btn.on{{background:#7c7cff;color:#fff;border-color:#7c7cff}}
.stats{{font-size:11px;color:#888}}
#overview{{display:block;width:100vw;height:calc(100vh - 44px)}}
#overview svg{{width:100%;height:100%}}
.link{{stroke-opacity:.15}}
.node circle{{stroke:#333;stroke-width:1;cursor:pointer}}
.node text{{font-size:9px;fill:#999;pointer-events:none}}
#focus{{display:none;height:calc(100vh - 44px)}}
.fc{{display:flex;height:100%}}
.col{{flex:1;display:flex;flex-direction:column;padding:12px;overflow-y:auto}}
.col.L{{background:#12121f;border-right:1px solid #2a2a4a}}
.col.C{{background:#1a1a30;flex:0 0 340px;align-items:center;justify-content:center}}
.col.R{{background:#12121f;border-left:1px solid #2a2a4a}}
.ch{{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px;
  margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #2a2a4a}}
.ch b{{color:#7c7cff}}
.cd{{padding:8px 10px;margin:3px 0;border-radius:5px;cursor:pointer;
  border:1px solid transparent;transition:all .15s}}
.cd:hover{{background:#22224a;border-color:#3a3a6a}}
.cd .fn{{font-size:12px;font-weight:600;color:#d0d0ff}}
.cd .ff{{font-size:10px;color:#666;margin-top:2px}}
.cd .fs{{font-size:10px;color:#888;margin-top:2px;font-style:italic;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.cd .fk{{font-size:9px;color:#555;background:#1a1a2e;padding:1px 5px;
  border-radius:3px;display:inline-block;margin-top:3px}}
.sf{{text-align:center}}
.sf .fn{{font-size:18px;color:#7c7cff;font-weight:700}}
.sf .ff{{font-size:12px;color:#888;margin-top:6px}}
.sf .fs{{font-size:12px;color:#aaa;margin-top:8px;font-family:monospace}}
.sf .fk{{font-size:11px;margin-top:8px}}
.ar{{text-align:center;padding:20px 0}}
.ar span{{font-size:28px;color:#3a3a6a}}
</style>
</head>
<body>
<div class="topbar">
  <h1>Call Graph Explorer</h1>
  <div class="search-box"><input id="q" placeholder="Search functions..."/></div>
  <button class="btn on" id="b1" onclick="M(1)">Overview</button>
  <button class="btn" id="b2" onclick="M(2)">Focus</button>
  <span class="stats" id="st"></span>
</div>
<div id="overview"><svg></svg></div>
<div id="focus"><div class="fc">
  <div class="col L">
    <div class="ch">Callers <b id="cc"></b></div><div id="cl"></div>
  </div>
  <div class="col C">
    <div class="ar"><span>←</span></div>
    <div class="sf" id="sf"></div>
    <div class="ar"><span>→</span></div>
  </div>
  <div class="col R">
    <div class="ch">Callees <b id="ec"></b></div><div id="el"></div>
  </div>
</div></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const N=[{nodes}];
const E=[{links}];
const G=[{groups}];

// Expand compact nodes: add id, expand link arrays
N.forEach((d,i)=>d.id=i);
const L=E.map(e=>({{source:e[0],target:e[1],via:e[2]==='c'?'calls':'types'}}));

document.getElementById('st').textContent=N.length+' functions, '+L.length+' edges';

// Adjacency
const CR={{}},CE={{}};
N.forEach(d=>{{CR[d.id]=[];CE[d.id]=[];}});
L.forEach(l=>{{
  CE[l.source].push({{t:l.target,v:l.via}});
  CR[l.target].push({{s:l.source,v:l.via}});
}});

const col=d3.scaleOrdinal(d3.schemeTableau10).domain(d3.range(G.length));
let sel=null;

// Overview
const svg=d3.select('#overview svg'),W=innerWidth,H=innerHeight-44,
  gg=svg.append('g');
svg.call(d3.zoom().scaleExtent([.05,10]).on('zoom',e=>gg.attr('transform',e.transform)));

const sim=d3.forceSimulation(N)
  .force('link',d3.forceLink(L).id(d=>d.id).distance(50).strength(.3))
  .force('charge',d3.forceManyBody().strength(-60))
  .force('center',d3.forceCenter(W/2,H/2))
  .force('collision',d3.forceCollide(12));

svg.append('defs').append('marker').attr('id','a').attr('viewBox','0 -4 8 8')
  .attr('refX',14).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto')
  .append('path').attr('d','M0,-4L8,0L0,4').attr('fill','#e94560').attr('opacity',.5);

const lk=gg.append('g').selectAll('line').data(L).join('line')
  .attr('class','link')
  .attr('stroke',d=>d.via==='types'?'#1a3a6a':'#3a2a6a')
  .attr('stroke-width',.5).attr('marker-end','url(#a)');

const nd=gg.append('g').selectAll('g').data(N).join('g').attr('class','node')
  .call(d3.drag()
    .on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}})
    .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));

nd.append('circle')
  .attr('r',d=>3+Math.sqrt((CR[d.id]?.length||0)+(CE[d.id]?.length||0)))
  .attr('fill',d=>col(d.g));
nd.append('text').attr('dx',8).attr('dy',3).text(d=>d.n.split('::').pop());

nd.on('click',(e,d)=>{{e.stopPropagation();S(d.id);M(2);}});
svg.on('click',()=>{{
  nd.select('circle').attr('opacity',1);
  nd.select('text').attr('opacity',1);
  lk.attr('stroke-opacity',.15);
}});

sim.on('tick',()=>{{
  lk.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
    .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  nd.attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
}});

// Search
const qi=document.getElementById('q');
qi.addEventListener('input',e=>{{
  const q=e.target.value.toLowerCase();
  if(!q){{nd.select('circle').attr('opacity',1);nd.select('text').attr('opacity',1);return;}}
  nd.select('circle').attr('opacity',d=>d.n.toLowerCase().includes(q)?1:.05);
  nd.select('text').attr('opacity',d=>d.n.toLowerCase().includes(q)?1:.02);
}});
qi.addEventListener('keydown',e=>{{
  if(e.key==='Enter'){{
    const q=e.target.value.toLowerCase();
    const m=N.find(d=>d.n.toLowerCase().includes(q));
    if(m){{S(m.id);M(2);}}
  }}
}});

// Mode
function M(m){{
  document.getElementById('overview').style.display=m===1?'block':'none';
  document.getElementById('focus').style.display=m===2?'block':'none';
  document.getElementById('b1').className='btn'+(m===1?' on':'');
  document.getElementById('b2').className='btn'+(m===2?' on':'');
  if(m===2&&sel===null&&N.length)S(0);
}}

// Select function
function S(id){{
  sel=id;const d=N[id];if(!d)return;

  document.getElementById('sf').innerHTML=
    `<div class="fn">${{d.n}}</div><div class="ff">${{d.f}}${{d.l?':'+d.l:''}}</div>`+
    (d.s?`<div class="fs">${{d.s}}</div>`:'')+`<div class="fk">${{d.k}}</div>`;

  const cr=CR[id]||[],ce=CE[id]||[];
  document.getElementById('cc').textContent='('+cr.length+')';
  document.getElementById('ec').textContent='('+ce.length+')';

  const cl=document.getElementById('cl');cl.innerHTML='';
  [...cr].sort((a,b)=>N[a.s].n.localeCompare(N[b.s].n)).forEach(c=>{{
    const cd=K(N[c.s],c.v);cd.onclick=()=>S(c.s);cl.appendChild(cd);
  }});

  const el=document.getElementById('el');el.innerHTML='';
  [...ce].sort((a,b)=>N[a.t].n.localeCompare(N[b.t].n)).forEach(c=>{{
    const cd=K(N[c.t],c.v);cd.onclick=()=>S(c.t);el.appendChild(cd);
  }});

  // Highlight
  const nb=new Set([id]);cr.forEach(c=>nb.add(c.s));ce.forEach(c=>nb.add(c.t));
  nd.select('circle').attr('opacity',d=>nb.has(d.id)?1:.08);
  nd.select('text').attr('opacity',d=>nb.has(d.id)?1:.03);
  lk.attr('stroke-opacity',l=>{{
    const s=typeof l.source==='object'?l.source.id:l.source;
    const t=typeof l.target==='object'?l.target.id:l.target;
    return(s===id||t===id)?.7:.02;
  }});
}}

function K(d,v){{
  const e=document.createElement('div');e.className='cd';
  e.innerHTML=`<div class="fn">${{d.n}}</div><div class="ff">${{d.f}}${{d.l?':'+d.l:''}}</div>`+
    (d.s?`<div class="fs">${{d.s}}</div>`:'')+
    `<span class="fk">${{d.k}}</span> <span class="fk" style="color:${{v==='types'?'#4a7aff':'#9a5aff'}}">${{v}}</span>`;
  return e;
}}
</script>
</body>
</html>"##,
        nodes = nodes_json,
        links = links_json,
        groups = groups_json,
    )
}
