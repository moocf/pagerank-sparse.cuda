const fs = require('fs');
const os = require('os');

const RGRAPH  = /Loading\s+graph\s+(\S+\/([^\/]*?).mtx)\s+\.\.\./;
const RORDER =  /order:\s*(\d+)\s+size:\s*(\d+)/;
const RRESULT = /\[(\S*)\s+ms\]\s+\[(\S*)\]\s+(\S*)(?:\s+\{(\S*)\}\s+\{(.*)\})?/;




function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}


function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}


function writeCsv(pth, rs, cs) {
  var cs = cs||Object.keys(rs[0]);
  var a  = cs.join()+'\n';
  for (var r of rs) {
    for (var c of cs)
      a += `${r[c]},`;
    a = a.substring(0, a.length-1)+'\n';
  }
  writeFile(pth, a);
}


function resultConfig(fn, mode, flags) {
  if (fn === 'pageRank')        return 'cpu';
  if (fn === 'pageRankNvgraph') return 'nvgraph';
  if (fn === 'pageRankCuda')    return `${mode} {${flags.trim()}}`;
  return `s-${mode} {${flags.trim()}}`;
}


function parseLog(pth) {
  var d = readFile(pth);
  var ls = d.split(/\n/g).map(l => l.trim());
  var a = new Map(), g = null;
  for (var l of ls) {
    if (RGRAPH.test(l)) {
      var [,, name] = l.match(RGRAPH);
      g = {name};
    }
    else if (RORDER.test(l)) {
      var [, order, size] = l.match(RORDER);
      g.order = parseFloat(order);
      g.size  = parseFloat(size);
    }
    else if (RRESULT.test(l)) {
      var [, time, error, fn, mode, flags] = l.match(RRESULT);
      var r      = {};
      r.graph    = g.name;
      r.vertices = g.order;
      r.edges    = g.size;
      r.config   = resultConfig(fn, mode, flags);
      r.error    = parseFloat(error);
      r.time     = parseFloat(time);
      if (!a.has(r.graph)) a.set(r.graph, []);
      a.get(r.graph).push(r);
    }
  }
  return a;
}


function postProcess(m) {
  var a = [];
  for (var rs of m.values()) {
    var time_nvgraph = rs.find(r => r.config==='nvgraph').time;
    for (var r of rs) {
      r.time_nvgraph = time_nvgraph;
      r.speedup      = time_nvgraph/r.time;
      r.speedupf     = r.speedup * (r.vertices + r.edges);
    }
    rs.sort((r, s) => r.time - s.time);
    a.push(...rs);
  }
  return a;
}


function main(a) {
  var [,, log, csv] = a;
  var rs = parseLog(log);
  var rs = postProcess(rs);
  writeCsv(csv, rs);
}
main(process.argv);
