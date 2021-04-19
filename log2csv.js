const fs = require('fs');
const os = require('os');
const path = require('path');

const RGRAPH  = /Loading\s+graph\s+(\S+\/([^\/]*?).mtx)\s+\.\.\./;
const RORDER = /^order:\s*(\d+)\s+size:\s*(\d+)/;
const RRESULT = /\[(\S*)\s+ms\]\s+\[(\S*)\]\s+(\S*)(?:\s+\{(\S*)\}\s+\{(.*)\})?/;




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
    var name = r.name;
    var time_nvgraph = parseInt(r.time_nvgraph, 10);
    a.set(name, {name, time_nvgraph});
  }
  return a;
}


function resultConfig(fn, mode, flags) {
  if (fn === 'pageRank') return 'cpu';
  if (fn === 'pageRankCuda') return `${mode} {${flags.trim()}}`;
  return `s-${mode} {${flags.trim()}}`;
}


function parseLog(m, pth) {
  var d = readFile(pth);
  var ls = d.split(/\n/g).map(l => l.trim());
  var a = new Map(), g = null;
  for (var l of ls) {
    if (RGRAPH.test(l)) {
      var [,, name] = l.match(RGRAPH);
      g = m.get(name);
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
      r.config   = resultConfig(fn, mode, flags);
      r.error    = parseFloat(error);
      r.time     = parseFloat(time);
      r.speedup  = r.time/g.time_nvgraph;
      r.speedupf = r.speedup * (g.order + g.size);
      if (!a.has(r.graph)) a.set(r.graph, []);
      a.get(r.graph).push(r);
    }
  }
  return a;
}


function postProcess(m) {
  var a = [];
  for (var rs of m.values()) {
    rs.sort((r, s) => r.time - s.time);
    a.push(...rs);
  }
  return a;
}


function main(a) {
  var [,, logfile, csvfile] = a;
  var reffile = path.join(__dirname, 'reference.csv');
  var ref = readCsv(reffile);
  var grp = parseReference(ref);
  var log = parseLog(grp, logfile);
  var rs  = postProcess(log);
  writeCsv(csvfile, rs);
}
main(process.argv);
