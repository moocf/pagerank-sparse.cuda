const fs = require('fs');
const os = require('os');
const path = require('path');

const RGRAPH  = /Loading\s+graph\s+(\S+\/([^\/]*?).mtx)\s+\.\.\./;
const RRESULT = /\[(\S*)\s+ms\]\s+\[(\S*)\]\s+(\S*)(?:\s+\{(\S*)\}\s+\{(.*)\})?/;

var KEYS = new Set();




function makeObject(ks, vs) {
  var a = {};
  for (var i=0, I=ks.length; i<I; i++)
    a[ks[i]] = vs[i];
  return a;
}


function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}


function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}


function readCsv(pth) {
  var d = readFile(pth);
  var ls = d.split(/\n/g).map(l => l.trim());
  var cs = ls[0].split(','), a = [];
  for (var l of ls.slice(1)) {
    if (!l) continue;
    var r = makeObject(cs, l.split(','));
    a.push(r);
  }
  return a;
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


function parseReference(rs) {
  var a = new Map();
  for (var r of rs) {
    var name       = r.name;
    var vertices   = parseInt(r.vertices, 10);
    var edges      = parseInt(r.edges, 10);
    var components = parseInt(r.components, 10);
    var levels     = parseInt(r.levels, 10);
    var nvgraph    = parseInt(r.nvgraph, 10);
    a.set(name, {name, vertices, edges, components, levels, nvgraph});
  }
  return a;
}


function parseLog(m, pth) {
  var d = readFile(pth);
  var ls = d.split(/\n/g).map(l => l.trim());
  var r = null;
  for (var l of ls) {
    if (RGRAPH.test(l)) {
      var [,, name] = l.match(RGRAPH);
      r = m.get(name);
    }
    else if (RRESULT.test(l)) {
      var [, time, err, fn, mode, flags] = l.match(RRESULT);
      var k = fn === 'pageRank'? 'cpu' : `${mode} {${flags.trim()}}`;
      var t = parseFloat(time), e   = parseFloat(err);
      var s = t/r.nvgraph, sve = s * (r.vertices + r.edges);
      r[`${k} time`]  = t;
      r[`${k} error`] = e;
      r[`${k} speedup (this/nvgraph)`] = s;
      r[`${k} speedup * (V+E)`]        = sve;
      KEYS.add(k);
    }
  }
  return m;
}


function bestKey(r) {
  var l = null;
  for (var k of KEYS)
    if (!l || r[`${k} time`] < r[`${l} time`]) l = k;
  return l;
}


function postProcess(m) {
  for (var r of m.values()) {
    var k = bestKey(r);
    r[`best mode`] = k;
    r[`best time`] = r[`${k} time`];
    r[`best error`] = r[`${k} error`];
    r[`best speedup (this/nvgraph)`] = r[`${k} speedup (this/nvgraph)`];
    r[`best speedup * (V+E)`]  = r[`${k} speedup * (V+E)`];
  }
}


function orderKeys() {
  var a = ['name', 'vertices', 'edges', 'components', 'levels', 'nvgraph'];
  a.push('best mode', 'best time', 'best error', 'best speedup (this/nvgraph)', 'best speedup * (V+E)');
  for (var k of KEYS)
    a.push(`${k} time`);
  for (var k of KEYS)
    a.push(`${k} error`);
  for (var k of KEYS)
    a.push(`${k} speedup (this/nvgraph)`);
  for (var k of KEYS)
    a.push(`${k} speedup * (V+E)`);
  return a;
}


function main(a) {
  var [,, log, csv] = a;
  var reference = path.join(__dirname, 'reference.csv');
  var rs = readCsv(reference);
  var m  = parseReference(rs);
  parseLog(m, log);
  postProcess(m);
  var cs = orderKeys();
  writeCsv(csv, [...m.values()], cs);
}
main(process.argv);
