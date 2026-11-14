from collections import defaultdict
from datetime import date,timedelta
from fractions import Fraction
import re
from .common import require,ident,number,rows
from .forecast import predict,metrics,ratio

def analyze(data):
    require(data,'observations initial_train horizon mean_window season');series=defaultdict(dict)
    initial=number(data['initial_train'],2,10000);horizon=number(data['horizon'],1,100)
    window=number(data['mean_window'],1,initial);season=number(data['season'],1,initial)
    for r in rows(data['observations'],20000):
        require(r,'series date quantity');sid=ident(r['series']);day=r['date']
        if not isinstance(day,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',day):raise ValueError('date must be YYYY-MM-DD')
        day=date.fromisoformat(day);value=number(r['quantity'])
        if day in series[sid]:raise ValueError('duplicate series date')
        series[sid][day]=value
    if not series:raise ValueError('at least one series required')
    count=sum(max(0,len(values)-initial-horizon+1)*horizon*3 for values in series.values())
    if count>200000:raise ValueError('evaluation exceeds 200000 forecast points')
    points=[];origins=[];grouped=defaultdict(list)
    for sid,values in sorted(series.items()):
        days=sorted(values)
        if len(days)<initial+horizon:raise ValueError('series needs a complete training and evaluation window')
        if any(b-a!=timedelta(days=1) for a,b in zip(days,days[1:])):raise ValueError('missing calendar dates are not zero demand')
        ordered=[values[d] for d in days]
        for cut in range(initial,len(days)-horizon+1):
            train=ordered[:cut];scale=Fraction(sum(abs(b-a) for a,b in zip(train,train[1:])),len(train)-1)
            predictions=predict(train,horizon,window,season)
            origins.append({'series':sid,'train_end':days[cut-1].isoformat(),'training_count':cut,'mase_scale':ratio(scale)})
            for method,forecasts in sorted(predictions.items()):
                for h,forecast in enumerate(forecasts,1):
                    point={'series':sid,'origin':days[cut-1].isoformat(),'target_date':days[cut+h-1].isoformat(),
                           'horizon':h,'method':method,'forecast':forecast,'actual':ordered[cut+h-1],'scale':scale}
                    points.append(point);grouped[sid,method].append(point)
    if len(points)>200000:raise ValueError('evaluation exceeds 200000 forecast points')
    summaries=[]
    for sid in sorted(series):
        for method in ('moving_mean','naive','seasonal_naive'):
            group=grouped[sid,method]
            summaries.append(dict(series=sid,method=method,**metrics(group)))
    serial=[dict(p,forecast=ratio(p['forecast']),scale=ratio(p['scale']),error_forecast_minus_actual=ratio(p['forecast']-p['actual'])) for p in points]
    return {'origins':origins,'forecasts':serial,'metrics':summaries,
            'evaluation_unit':'origin-target forecast; overlapping targets are intentionally counted separately'}
