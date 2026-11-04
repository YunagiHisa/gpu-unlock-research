"""Render public comparison CSVs as deterministic SVGs; standard library only."""
import argparse
import csv
import html
import math
from pathlib import Path

PHASES = ('before', 'after', 'reverted')
COLORS = {'before': '#526d82', 'after': '#07856f', 'reverted': '#c77816'}


class SVG:
    def __init__(self, width, height, title, description):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
                      f'<title>{html.escape(title)}</title><desc>{html.escape(description)}</desc>',
                      '<rect width="100%" height="100%" fill="#ffffff"/>']
        self.text(32, 36, title, size=23, weight='bold')

    def text(self, x, y, value, size=13, anchor='start', color='#243447', weight='normal'):
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{html.escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color='#dce3e9', width=1):
        self.parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, x, y, width, height, color, attributes=None):
        attrs = ''.join(f' {k}="{html.escape(str(v), quote=True)}"' for k, v in (attributes or {}).items())
        self.parts.append(f'<rect x="{x:.6f}" y="{y:.6f}" width="{width:.6f}" height="{height:.6f}" fill="{color}"{attrs}/>')

    def save(self, path):
        path.write_text('\n'.join(self.parts+['</svg>'])+'\n')


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def legend(svg, x=32, y=68):
    for index, phase in enumerate(PHASES):
        xx = x+index*140
        svg.rect(xx, y-11, 16, 12, COLORS[phase])
        svg.text(xx+23, y, phase.title())


def bf16_chart(data, output):
    chart = SVG(960, 520, 'BF16 GEMM: Before, After and Reverted',
                '8192 cubed matrices on two RTX 3060 GPUs. Means and sample SDs of three independent processes per phase.')
    legend(chart)
    left, top, width, height = 90, 110, 810, 310
    y = lambda value: top+height-value/55*height
    for tick in range(0, 56, 10):
        chart.line(left, y(tick), left+width, y(tick))
        chart.text(left-12, y(tick)+4, tick, anchor='end')
    chart.text(left, top-15, 'TFLOPS')
    for gpu in (0, 1):
        row = next(r for r in data if r['gpu'] == str(gpu) and r['precision'] == 'bf16' and r['matrix_size'] == '8192')
        center = left+width*(0.25+0.5*gpu)
        for index, phase in enumerate(PHASES):
            mean, sd = float(row[phase+'_mean']), float(row[phase+'_sd'])
            x = center+(index-1)*82-30
            chart.rect(x, y(mean), 60, y(0)-y(mean), COLORS[phase],
                       {'class': 'data-bar', 'data-kind': 'bf16', 'data-gpu': gpu, 'data-phase': phase,
                        'data-value': repr(mean), 'data-sd': repr(sd)})
            chart.line(x+30, y(mean-sd), x+30, y(mean+sd), '#182b38', 1.5)
            for val in (mean-sd, mean+sd):
                chart.line(x+22, y(val), x+38, y(val), '#182b38', 1.5)
            chart.text(x+30, y(mean+sd)-10, f'{mean:.3f}', anchor='middle')
        chart.text(center, top+height+28, f'GPU {gpu} · 8192³', size=16, anchor='middle')
    chart.text(32, 478, 'Bars: mean of 3 process TFLOPS values; whiskers: ±1 sample SD across processes.')
    chart.text(32, 500, 'Preliminary and smoke runs excluded. Same kernel, driver and measurement code across phases.', size=12)
    chart.save(output/'gpu-unlock-bf16-8192.svg')


def gain_chart(data, output):
    chart = SVG(1120, 820, 'GEMM improvement by GPU, precision and matrix size',
                'After versus Before ratios of three-process means, covering all 24 GEMM configurations.')
    size_colors = {'2048': '#6c83b5', '4096': '#15877a', '8192': '#c77816'}
    for index, size in enumerate(size_colors):
        x = 32+index*140
        chart.rect(x, 58, 16, 12, size_colors[size])
        chart.text(x+23, 70, size+'³')
    left, width, height = 88, 990, 240
    for gpu in (0, 1):
        top = 115+gpu*335
        y = lambda value: top+height-(value+5)/95*height
        chart.text(32, top-12, f'GPU {gpu}', size=18, weight='bold')
        for tick in (0, 20, 40, 60, 80):
            chart.line(left, y(tick), left+width, y(tick))
            chart.text(left-10, y(tick)+4, f'{tick}%', anchor='end')
        chart.line(left, y(0), left+width, y(0), '#6a7883')
        for group, mode in enumerate(('fp16', 'bf16', 'tf32', 'fp32')):
            center = left+width*(group+0.5)/4
            for index, size in enumerate(size_colors):
                r = next(r for r in data if r['gpu'] == str(gpu) and r['precision'] == mode and r['matrix_size'] == size)
                value = float(r['after_vs_before_percent'])
                x = center+(index-1)*53-21
                chart.rect(x, min(y(value), y(0)), 42, abs(y(value)-y(0)), size_colors[size],
                           {'class': 'data-bar', 'data-kind': 'gain', 'data-gpu': gpu,
                            'data-mode': mode, 'data-size': size, 'data-value': repr(value)})
                chart.text(x+21, y(value)-8 if value >= 0 else y(0)+18, f'{value:+.2f}', size=11, anchor='middle')
            chart.text(center, top+height+40, 'FP32 IEEE' if mode == 'fp32' else mode.upper(), anchor='middle', size=15)
    chart.text(32, 765, 'Bars: 100 × (After mean / Before mean − 1). No inferential error bars; process SDs are in the CSV.')
    chart.text(32, 790, 'FP32 IEEE: −0.24% to −0.01%; its small negative bars sit near zero, not at exactly zero.', size=12)
    chart.save(output/'gpu-unlock-gemm-gains.svg')


def llama_chart(data, output):
    chart = SVG(1120, 800, 'llama.cpp: prefill and token generation',
                'Three-phase means normalized to Before. Error bars show process SD divided by the fixed Before mean, not confidence intervals.')
    legend(chart)
    left, width, xmax = 300, 680, 1.25
    x = lambda value: left+value/xmax*width
    ordered = [('Qwen3.8-27B', 'Q4_K_M', 'pp512'), ('Qwen3.8-27B', 'Q4_K_M', 'pp2048'),
               ('Qwen3-8B', 'Q4_K_M', 'pp512'), ('Qwen3-8B', 'BF16', 'pp512'),
               ('Qwen3.8-27B', 'Q4_K_M', 'tg128'), ('Qwen3-8B', 'Q4_K_M', 'tg128'), ('Qwen3-8B', 'BF16', 'tg128')]
    for tick in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
        chart.line(x(tick), 105, x(tick), 697)
        chart.text(x(tick), 720, f'{tick:g}×', anchor='middle')
    chart.line(x(1), 105, x(1), 697, '#6a7883', 1.5)
    chart.text(32, 110, 'Prefill', size=17, weight='bold')
    chart.text(32, 451, 'Token generation', size=17, weight='bold')
    for index, (model, quant, test) in enumerate(ordered):
        r = next(r for r in data if r['model'] == model and r['quantization'] == quant and r['benchmark'] == test)
        yy = 148+index*76+(37 if index >= 4 else 0)
        chart.text(left-16, yy+5, f'{model} {quant}', anchor='end')
        chart.text(left-16, yy+22, test, size=12, anchor='end', color='#576776')
        denominator = float(r['before_mean'])
        for pi, phase in enumerate(PHASES):
            mean, sd = float(r[phase+'_mean']), float(r[phase+'_sd'])
            value, error = mean/denominator, sd/denominator
            y = yy+(pi-1)*15
            chart.rect(x(0), y-5, x(value)-x(0), 10, COLORS[phase],
                       {'class': 'data-bar', 'data-kind': 'llama', 'data-model': model,
                        'data-quantization': quant, 'data-benchmark': test, 'data-phase': phase,
                        'data-value': repr(mean), 'data-sd': repr(sd), 'data-normalized': repr(value)})
            chart.line(x(value-error), y, x(value+error), y, '#182b38', 1.5)
            for val in (value-error, value+error):
                chart.line(x(val), y-4, x(val), y+4, '#182b38', 1.5)
        chart.text(1000, yy, f"A/B {float(r['after_vs_before_percent']):+.2f}%", size=12)
        chart.text(1000, yy+18, f"R/B {float(r['reverted_vs_before_percent']):+.2f}%", size=12)
    chart.text(32, 755, 'Bars: 3 independent process means, each process containing 5 repeats; Before mean = 1×.')
    chart.text(32, 778, 'Whiskers: process SD / fixed Before mean. Absolute tok/s and SDs are in the CSV.', size=12)
    chart.save(output/'gpu-unlock-llama-comparison.svg')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('docs/data'))
    p.add_argument('--output-dir', type=Path, default=Path('docs/images'))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    gemm = rows(a.data_dir/'gpu-unlock-gemm-comparison.csv')
    llama = rows(a.data_dir/'gpu-unlock-llama-comparison.csv')
    assert len(gemm) == 24 and len(llama) == 7
    bf16_chart(gemm, a.output_dir)
    gain_chart(gemm, a.output_dir)
    llama_chart(llama, a.output_dir)
    print('Rendered 3 SVG figures from public comparison CSVs using only the Python standard library.')


if __name__ == '__main__':
    main()
