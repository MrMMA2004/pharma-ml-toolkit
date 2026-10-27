import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync } from 'node:fs';

async function main() {
  const shots: Array<[string, string, string]> = [
    [
      'pg154-ml-shap-beeswarm.png',
      'A desktop ML toolkit screenshot (the "Data & ML" tab, the Results card). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The Results card shows a SHAP BEESWARM: three horizontal feature rows labelled "strong", "weak" and "noise" (strong at the TOP), each row a band of small coloured dots spreading left and right of a central dashed zero line.\n' +
        '2) The dots are coloured on a ramp from a warm/amber tone to a teal/green tone, and a colour bar legend on the right reads "feature value (row-normalised)".\n' +
        '3) The x-axis label mentions "SHAP value (impact on model output" and a base value, and the figure title mentions "SHAP beeswarm" and "held-out rows".\n' +
        '4) A button row includes buttons reading "SHAP beeswarm" and "SHAP top-20 bar", both fully visible.\n' +
        '5) No overlapping, clipped, or garbled text anywhere in the window.\n',
      'vlm154_shap_beeswarm.json',
    ],
    [
      'pg154-ml-shap-top20-bar.png',
      'A desktop ML toolkit screenshot (the "Data & ML" tab, the Results card). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The Results card shows a horizontal BAR chart with exactly three bars labelled "strong", "weak" and "noise" (strong the longest, at the top), each carrying a numeric value annotation at its end.\n' +
        '2) The x-axis is labelled "mean |SHAP value|" and the title mentions "SHAP importance".\n' +
        '3) A button row includes buttons reading "SHAP beeswarm" and "SHAP top-20 bar", both fully visible.\n' +
        '4) The note line under the metrics mentions "top-20 bar", "held-out row(s)" and a base value.\n' +
        '5) No overlapping, clipped, or garbled text anywhere in the window.\n',
      'vlm154_shap_top20.json',
    ],
  ];
  const zai = await ZAI.create();
  for (const [name, prompt, out] of shots) {
    const data = readFileSync(`download/${name}`);
    const url = `data:image/png;base64,${data.toString('base64')}`;
    const response = await zai.chat.completions.createVision({
      model: 'glm-5v-turbo',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url } },
          ],
        },
      ],
    });
    writeFileSync(`scratch/${out}`, JSON.stringify(response, null, 2));
    console.log(`${name}: ${(response as any).choices[0].message.content.split('\n').join(' | ')}`);
  }
}

main().catch((error) => {
  console.error('VLM review failed:', error);
  process.exit(1);
});
