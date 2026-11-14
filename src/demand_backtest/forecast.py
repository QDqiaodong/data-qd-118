"""Forecast functions accept only the training prefix, never held-out targets."""
from fractions import Fraction

def predict(train,horizon,window,season):
    return {'naive':[Fraction(train[-1])]*horizon,
            'moving_mean':[Fraction(sum(train[-window:]),window)]*horizon,
            'seasonal_naive':[Fraction(train[-season+(h%season)]) for h in range(horizon)]}

def ratio(x):return str(x) if x is not None else None

def metrics(points):
    n=len(points)
    if not n:
        return {'count':0,'mae':None,'mse':None,'bias_forecast_minus_actual':None,
                'wape':None,'mase':None,'mase_count':0}
    absolute=sum((abs(p['actual']-p['forecast']) for p in points),Fraction())
    squared=sum(((p['actual']-p['forecast'])**2 for p in points),Fraction())
    signed=sum((p['forecast']-p['actual'] for p in points),Fraction())
    actual=sum(p['actual'] for p in points)
    scaled=[abs(p['actual']-p['forecast'])/p['scale'] for p in points if p['scale']!=0]
    return {'count':n,'mae':ratio(absolute/n),'mse':ratio(squared/n),'bias_forecast_minus_actual':ratio(signed/n),
            'wape':ratio(absolute/actual) if actual else None,
            'mase':ratio(sum(scaled,Fraction())/len(scaled)) if scaled else None,'mase_count':len(scaled)}
