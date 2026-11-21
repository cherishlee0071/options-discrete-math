"""Build an offline American put demo and verify the stopping distribution."""

import argparse
import csv
import json
import math
import statistics
from datetime import date, datetime
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / 'USmarket' / 'BRK.B_price.csv'
SNAPSHOT = ROOT / 'data/brkb_window.csv'
VALUATION_DATE = date(2026, 10, 22)
TREASURY_URL = ('https://home.treasury.gov/resource-center/data-chart-center/'
                'interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202610')
CRR_URL = 'https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/36891/files/2017/07/CRR79-1yy8av8.pdf'


def stock_window(path: Path) -> list[tuple[date, float]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = csv.DictReader(f)
        if {'Date', 'Close/Last'} <= set(rows.fieldnames or []):
            observations = [(datetime.strptime(row['Date'], '%m/%d/%Y').date(),
                             float(row['Close/Last'].replace('$', '').replace(',', '')))
                            for row in rows]
        else:
            observations = [(date.fromisoformat(row['date']), float(row['close'])) for row in rows]
    observations = sorted((day, close) for day, close in observations if day <= VALUATION_DATE)
    window = observations[-61:]
    if (len(window) != 61 or window[-1][0] != VALUATION_DATE or
            len({day for day, _ in window}) != 61 or
            any(not math.isfinite(close) or close <= 0 for _, close in window)):
        raise ValueError('Expected 61 distinct positive closes through 2026-10-22')
    return window


def treasury_rate(path: Path) -> float:
    with path.open(encoding='utf-8-sig', newline='') as f:
        matches = [float(row['3 Mo']) / 100 for row in csv.DictReader(f)
                   if datetime.strptime(row['Date'], '%m/%d/%Y').date() == VALUATION_DATE]
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise ValueError('Treasury CSV must contain one valid 3-month yield for 2026-10-22')
    return matches[0]


def price_tree(spot: float, strike: float, steps: int, up: float, gross: float) -> dict:
    down = 1 / up
    if steps < 1 or not down < gross < up:
        raise ValueError('Expected a positive step count and down < risk-free growth < up')
    p = (gross - down) / (up - down)
    layers = []
    for step in range(steps + 1):
        layer = []
        for ups in range(step + 1):
            stock = spot * up ** (2 * ups - step)
            layer.append({'stock': stock, 'payoff': max(strike - stock, 0), 'reach': 0.0})
        layers.append(layer)

    for node in layers[-1]:
        node['continuation'] = None
        node['value'] = node['payoff']
        node['action'] = 'maturity' if node['payoff'] > 0 else 'expire'
    for step in range(steps - 1, -1, -1):
        for ups, node in enumerate(layers[step]):
            next_layer = layers[step + 1]
            continuation = ((1 - p) * next_layer[ups]['value'] + p * next_layer[ups + 1]['value']) / gross
            node['continuation'] = continuation
            node['value'] = max(node['payoff'], continuation)
            # Continue at an exact tie, including zero-payoff/zero-value nodes.
            node['action'] = 'exercise' if node['payoff'] > continuation else 'continue'

    layers[0][0]['reach'] = 1.0
    distribution = []
    for step, layer in enumerate(layers):
        totals = {'step': step, 'reach': 0.0, 'early': 0.0, 'maturity': 0.0,
                  'expired': 0.0, 'remaining': 0.0, 'pv': 0.0}
        for ups, node in enumerate(layer):
            probability = node['reach']
            totals['reach'] += probability
            action = node['action']
            if action == 'continue':
                layers[step + 1][ups]['reach'] += probability * (1 - p)
                layers[step + 1][ups + 1]['reach'] += probability * p
                totals['remaining'] += probability
            elif action == 'expire':
                totals['expired'] += probability
            else:
                totals['early' if action == 'exercise' else 'maturity'] += probability
                totals['pv'] += probability * node['payoff'] / gross ** step
        distribution.append(totals)
    return {'spot': spot, 'strike': strike, 'steps': steps, 'up': up, 'down': down,
            'gross': gross, 'p': p, 'price': layers[0][0]['value'], 'layers': layers,
            'distribution': distribution}


def verify(model: dict) -> dict:
    """Independently enumerate every full path and stop at its first exercise."""
    steps, p = model['steps'], model['p']
    by_step = [{'early': 0.0, 'maturity': 0.0, 'expired': 0.0, 'pv': 0.0}
               for _ in range(steps + 1)]
    for path in product((0, 1), repeat=steps):
        count_up = sum(path)
        probability = p ** count_up * (1 - p) ** (steps - count_up)
        ups = 0
        for step in range(steps + 1):
            action = model['layers'][step][ups]['action']
            if action != 'continue':
                stock = model['spot'] * model['up'] ** ups * model['down'] ** (step - ups)
                payoff = max(model['strike'] - stock, 0)
                event = {'exercise': 'early', 'maturity': 'maturity', 'expire': 'expired'}[action]
                by_step[step][event] += probability
                by_step[step]['pv'] += probability * payoff / model['gross'] ** step
                break
            ups += path[step]

    def require_close(a, b, label):
        if not math.isclose(a, b, rel_tol=0, abs_tol=1e-10):
            raise AssertionError(f'{label}: {a} != {b}')

    survival = 1.0
    for actual, expected in zip(model['distribution'], by_step):
        for key, value in expected.items():
            require_close(actual[key], value, f'{key} at step {actual["step"]}')
        require_close(actual['reach'], survival, 'incoming probability')
        require_close(actual['reach'], sum(actual[k] for k in ('early', 'maturity', 'expired', 'remaining')),
                      'step probability conservation')
        survival = actual['remaining']
    forward = sum(row['pv'] for row in model['distribution'])
    terminal = {key: sum(row[key] for row in model['distribution'])
                for key in ('early', 'maturity', 'expired')}
    require_close(sum(terminal.values()), 1, 'total stopping probability')
    require_close(forward, model['price'], 'forward/backward price')
    if model['steps'] == 3:
        for key, expected in zip(terminal, (.25, .25, .50)):
            require_close(terminal[key], expected, f'hand check: {key}')
        require_close(model['price'], .25 * 36 / 1.025**2 + .25 * 20 / 1.025**3, 'hand check: price')
    return {**terminal, 'forward_price': forward, 'path_count': 2**steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock-file', type=Path, help='Read the original stock CSV and refresh the local window')
    args = parser.parse_args()
    window = stock_window(args.stock_file or (SNAPSHOT if SNAPSHOT.exists() else SOURCE))
    treasury = treasury_rate(ROOT / 'data/treasury_202610.csv')
    volatility = statistics.stdev(math.log(b[1] / a[1]) for a, b in zip(window, window[1:])) * math.sqrt(252)
    dt = (90 / 365) / 12
    brkb = price_tree(window[-1][1], 550, 12, math.exp(volatility * math.sqrt(dt)), math.exp(treasury * dt))
    hand = price_tree(100, 100, 3, 1.25, 1.025)
    for model in (brkb, hand):
        model['checks'] = verify(model)
    data = {
        'valuation_date': VALUATION_DATE.isoformat(), 'window_start': window[0][0].isoformat(),
        'volatility': volatility, 'rate': treasury, 'term_days': 90,
        'sources': {'treasury': TREASURY_URL, 'crr': CRR_URL}, 'models': {'brkb': brkb, 'hand': hand},
    }
    with SNAPSHOT.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'close'])
        writer.writerows(window)
    with (ROOT / 'exercise_distribution.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, ['scenario', *brkb['distribution'][0].keys()])
        writer.writeheader()
        for scenario, model in data['models'].items():
            writer.writerows({'scenario': scenario, **row} for row in model['distribution'])
    template = (ROOT / 'demo_template.html').read_text(encoding='utf-8')
    if template.count('__DEMO_DATA__') != 1:
        raise ValueError('Invalid HTML template data placeholder')
    (ROOT / 'demo.html').write_text(template.replace('__DEMO_DATA__', json.dumps(data, allow_nan=False)),
                                    encoding='utf-8')
    for scenario, model in data['models'].items():
        print(scenario, f'price={model["price"]:.8f}', model['checks'])


if __name__ == '__main__':
    main()
