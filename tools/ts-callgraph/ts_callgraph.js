#!/usr/bin/env node
/**
 * TypeScript/JavaScript call graph extractor
 * Outputs edges.jsonl + chunks.jsonl (same format as mir-callgraph)
 * No external dependencies — uses regex-based parsing.
 */

const fs = require('fs');
const path = require('path');

const IGNORED_DIRS = new Set([
  'node_modules', '.git', 'dist', 'build', '.next', '.nuxt',
  'coverage', '__pycache__', '.cache', 'target',
]);

function extractFromFile(filePath, projectRoot) {
  const chunks = [];
  const edges = [];

  let source;
  try {
    source = fs.readFileSync(filePath, 'utf-8');
  } catch {
    return { chunks, edges };
  }

  const relPath = path.relative(projectRoot, filePath).replace(/\\/g, '/');
  const moduleName = relPath.replace(/\.(ts|tsx|js|jsx|mjs|cjs)$/, '').replace(/\//g, '::');

  const lines = source.split('\n');

  // Track class context
  let currentClass = null;
  let braceDepth = 0;
  let classEndDepth = -1;

  // Regex patterns
  const classPattern = /^(\s*)(export\s+)?(abstract\s+)?class\s+(\w+)/;
  const funcPattern = /^(\s*)(export\s+)?(async\s+)?function\s+(\w+)\s*(\([^)]*\))/;
  const methodPattern = /^(\s*)(public|private|protected|static|async|get|set|\s)*\s*(\w+)\s*(\([^)]*\))/;
  const arrowPattern = /^(\s*)(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\(/;
  const callPattern = /(\w+(?:\.\w+)*)\s*\(/g;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;

    // Track brace depth
    for (const ch of line) {
      if (ch === '{') braceDepth++;
      if (ch === '}') {
        braceDepth--;
        if (braceDepth <= classEndDepth) {
          currentClass = null;
          classEndDepth = -1;
        }
      }
    }

    // Class detection
    const classMatch = line.match(classPattern);
    if (classMatch) {
      const [, , exportKw, , className] = classMatch;
      const qualName = moduleName + '::' + className;
      currentClass = className;
      classEndDepth = braceDepth - 1;

      // Find end line (simplified: next line with same or less indentation)
      let endLine = lineNum;
      let depth = 0;
      for (let j = i; j < lines.length; j++) {
        for (const ch of lines[j]) {
          if (ch === '{') depth++;
          if (ch === '}') depth--;
        }
        if (depth <= 0 && j > i) { endLine = j + 1; break; }
      }

      chunks.push({
        name: qualName,
        file: relPath,
        kind: 'class',
        start_line: lineNum,
        end_line: endLine,
        signature: line.trim().split('{')[0].trim(),
        visibility: exportKw ? 'pub' : '',
        is_test: false,
      });
      continue;
    }

    // Function detection
    const funcMatch = line.match(funcPattern);
    const arrowMatch = !funcMatch && line.match(arrowPattern);
    const methodMatch = !funcMatch && !arrowMatch && currentClass && line.match(methodPattern);

    let funcName = null;
    let sig = null;
    let vis = '';
    let kind = 'fn';

    if (funcMatch) {
      const [, , exportKw, , name] = funcMatch;
      funcName = currentClass ? `${currentClass}::${name}` : name;
      sig = line.trim().split('{')[0].trim();
      vis = exportKw ? 'pub' : '';
    } else if (arrowMatch) {
      const [, , exportKw, , name] = arrowMatch;
      funcName = currentClass ? `${currentClass}::${name}` : name;
      sig = line.trim().split('=>')[0].trim();
      vis = exportKw ? 'pub' : '';
    } else if (methodMatch) {
      const [, , modifiers, name] = methodMatch;
      if (name && !['if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'throw', 'import', 'from', 'class'].includes(name)) {
        funcName = `${currentClass}::${name}`;
        sig = line.trim().split('{')[0].trim();
        kind = 'method';
        vis = (modifiers || '').includes('private') ? '' : 'pub';
      }
    }

    if (funcName) {
      const qualName = moduleName + '::' + funcName;

      // Find function end
      let endLine = lineNum;
      let depth = 0;
      for (let j = i; j < lines.length; j++) {
        for (const ch of lines[j]) {
          if (ch === '{') depth++;
          if (ch === '}') depth--;
        }
        if (depth <= 0 && j > i) { endLine = j + 1; break; }
      }

      // Detect test
      const isTest = funcName.startsWith('test') || funcName.includes('test_')
        || /\b(it|describe|test|expect)\s*\(/.test(line)
        || relPath.includes('test') || relPath.includes('spec')
        || (i > 0 && /\b(it|test|describe)\b/.test(lines[i - 1]));

      chunks.push({
        name: qualName,
        file: relPath,
        kind,
        start_line: lineNum,
        end_line: endLine,
        signature: sig,
        visibility: vis,
        is_test: isTest,
      });

      // Extract calls from function body
      for (let j = i; j < Math.min(endLine, lines.length); j++) {
        const bodyLine = lines[j];
        let callMatch2;
        callPattern.lastIndex = 0;
        while ((callMatch2 = callPattern.exec(bodyLine)) !== null) {
          const callee = callMatch2[1];
          // Skip keywords and common non-function calls
          if (['if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'throw', 'import', 'require', 'console', 'Math', 'Object', 'Array', 'String', 'Number', 'Boolean', 'JSON', 'Promise', 'Date', 'Error', 'RegExp', 'Map', 'Set', 'typeof', 'instanceof'].includes(callee.split('.')[0])) continue;

          edges.push({
            caller: qualName,
            caller_file: relPath,
            caller_kind: kind,
            callee: callee.replace(/\./g, '::'),
            line: j + 1,
            is_local: true,  // Will be refined
          });
        }
      }
    }
  }

  return { chunks, edges };
}

function walkDir(dir, projectRoot, allChunks, allEdges) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (IGNORED_DIRS.has(entry.name)) continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkDir(fullPath, projectRoot, allChunks, allEdges);
    } else if (/\.(ts|tsx|js|jsx|mjs|cjs)$/.test(entry.name)) {
      const { chunks, edges } = extractFromFile(fullPath, projectRoot);
      allChunks.push(...chunks);
      allEdges.push(...edges);
    }
  }
}

function main() {
  const args = process.argv.slice(2);
  if (args.length < 1) {
    console.error('Usage: ts_callgraph.js <project_root> [--out-dir <dir>]');
    process.exit(1);
  }

  const projectRoot = path.resolve(args[0]);
  let outDir = null;
  const outIdx = args.indexOf('--out-dir');
  if (outIdx !== -1 && outIdx + 1 < args.length) {
    outDir = args[outIdx + 1];
  }

  const allChunks = [];
  const allEdges = [];
  walkDir(projectRoot, projectRoot, allChunks, allEdges);

  // Mark is_local
  const chunkNames = new Set(allChunks.map(c => c.name));
  const shortNames = new Map();
  for (const c of allChunks) {
    const parts = c.name.split('::');
    shortNames.set(parts[parts.length - 1], c.name);
  }

  for (const edge of allEdges) {
    edge.is_local = chunkNames.has(edge.callee) || shortNames.has(edge.callee);
  }

  if (outDir) {
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, 'typescript.edges.jsonl'),
      allEdges.map(e => JSON.stringify(e)).join('\n') + '\n');
    fs.writeFileSync(path.join(outDir, 'typescript.chunks.jsonl'),
      allChunks.map(c => JSON.stringify(c)).join('\n') + '\n');
    console.error(`[ts-callgraph] ${allEdges.length} edges, ${allChunks.length} chunks`);
  } else {
    for (const e of allEdges) {
      console.log(JSON.stringify(e));
    }
  }
}

main();
