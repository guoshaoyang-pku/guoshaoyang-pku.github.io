"""Recompute the four-state zero-LL mean-field comparison of arXiv:1105.5386v2, p3."""
import csv, json, math, pathlib, time
BASE = pathlib.Path(__file__).resolve().parent
A, UZ, N = 1.0, 2.0, 100
COLORS = {'CAF': '#b6d3df', 'F': '#f3dc8c', 'PLP': '#b9cee9', 'FLP': '#dbbbb2', 'degenerate': '#ffffff'}
def energies(v, z):
    return {'F': 2*A-UZ-2*z, 'CAF': -UZ-z*z/(2*A) if z <= 2*A else math.inf,
            'PLP': -A-v*v/(UZ+A) if v <= UZ+A else math.inf, 'FLP': UZ-2*v}
def state(v, z):
    e = energies(v, z)
    minimum = min(e.values())
    tied = [p for p, value in e.items() if abs(value-minimum) <= 1e-10]
    return tied[0] if len(tied) == 1 else 'degenerate', tied, e
started = time.perf_counter()
rows = []
for j in range(N+1):
    z = 3.5*j/N
    for i in range(N+1):
        v = 4*i/N
        label, tied, e = state(v, z)
        rows.append({'electric_over_abs_uperp': v, 'zeeman_over_abs_uperp': z, 'phase': label, 'tied_phases': '|'.join(tied), **{p+'_energy': value if math.isfinite(value) else '' for p,value in e.items()}})
with (BASE/'kharitonov-four-state-map.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
validation = {'source': 'https://arxiv.org/abs/1105.5386v2', 'physical_pdf_page': 3, 'u_perp': -A, 'u_z': UZ, 'rows': len(rows), 'checks': []}
for name,v,z,expected in [('CAF interior',0,0,'CAF'),('F interior',0,3,'F'),('PLP interior',2.5,0,'PLP'),('FLP interior',4,0,'FLP')]:
    actual = state(v,z)[0]; assert actual == expected
    validation['checks'].append({'name':name,'expected':expected,'actual':actual})
for j in range(101):
    z = 2*j/100; v = math.sqrt(3*(1+z*z/2)); e = energies(v,z)
    assert abs(e['CAF']-e['PLP']) < 1e-12
for j in range(101):
    v = 3*j/100; e = energies(v,2); assert abs(e['F']-e['CAF']) < 1e-12
for j in range(101):
    z = 2+1.5*j/100; v = 1+z; e = energies(v,z); assert abs(e['F']-e['FLP']) < 1e-12
validation['analytic_boundary_checks'] = 303
validation['elapsed_seconds'] = time.perf_counter()-started
validation['scope'] = 'Zero-temperature, zero-Landau-level-projected four-candidate uniform mean field; no boundary in SI units and no experiment fitting.'
(BASE/'kharitonov-four-state-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
X,Y,W,H = 86,55,660,500
parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 865 690" role="img" aria-labelledby="title desc">','<title id="title">Four competing orders at filling ν=0</title>','<desc id="desc">Calculated mean-field comparison of CAF, ferromagnet, partial layer polarization and full layer polarization, following Kharitonov2012.</desc>','<rect width="865" height="690" fill="white"/>']
for j in range(N):
    for i in range(N):
        label = state(4*(i+.5)/N,3.5*(j+.5)/N)[0]
        parts.append(f'<rect x="{X+W*i/N:.3f}" y="{Y+H*(1-(j+1)/N):.3f}" width="{W/N+.1:.3f}" height="{H/N+.1:.3f}" fill="{COLORS[label]}"/>')
def point(v,z): return (X+W*v/4,Y+H*(1-z/3.5))
def path(points,dash=''):
    ps=' '.join(f'{x:.3f},{y:.3f}' for x,y in (point(v,z) for v,z in points))
    parts.append(f'<polyline points="{ps}" fill="none" stroke="#263746" stroke-width="2.1" stroke-dasharray="{dash}"/>')
path([(math.sqrt(3*(1+(2*j/100)**2/2)),2*j/100) for j in range(101)])
path([(0,2),(3,2)],'7 5'); path([(3,0),(3,2)],'7 5'); path([(3,2),(4,3)])
parts.append(f'<rect x="{X}" y="{Y}" width="{W}" height="{H}" fill="none" stroke="#263746"/>')
for v in range(5):
    x,_=point(v,0); parts.append(f'<text x="{x}" y="581" text-anchor="middle" font-family="Arial" font-size="16">{v}</text>')
for z in [0,1,2,3,3.5]:
    _,y=point(0,z); parts.append(f'<text x="70" y="{y+5}" text-anchor="end" font-family="Arial" font-size="16">{z}</text>')
for name,v,z in [('CAF',.75,.9),('F',1.2,2.8),('PLP',2.55,.9),('FLP',3.6,1.3)]:
    x,y=point(v,z); parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Arial" font-size="24" fill="#182b3a">{name}</text>')
parts += ['<text x="416" y="617" text-anchor="middle" font-family="Arial" font-size="18">Electric layer energy εV / |u⊥|</text>','<text x="26" y="305" text-anchor="middle" transform="rotate(-90 26 305)" font-family="Arial" font-size="18">Zeeman energy εZ / |u⊥|</text>','<text x="86" y="30" font-family="Arial" font-size="20">ν = 0 · four-state mean field · uz / |u⊥| = 2</text>','<text x="86" y="653" font-family="Arial" font-size="15">Solid: first-order candidate crossing. Dashed: continuous endpoint.</text>','<text x="86" y="676" font-family="Arial" font-size="13" fill="#51616d">Recomputed from Kharitonov, arXiv:1105.5386v2, p3. Model comparison, not an all-phase diagram.</text>','</svg>']
(BASE/'kharitonov-four-state-map.svg').write_text('\n'.join(parts)+'\n')
print(json.dumps(validation,indent=2))
