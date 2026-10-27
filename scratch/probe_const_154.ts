import { pearsonCorrelation } from "../src/lib/ml-math";
const big = Array.from({length: 38}, () => 1e300);
const y = Array.from({length: 38}, (_, i) => i);
console.log("constant 1e300 vs y:", pearsonCorrelation(big, y));
const tiny = Array.from({length: 38}, () => 1e-300);
console.log("constant 1e-300 vs y:", pearsonCorrelation(tiny, y));
