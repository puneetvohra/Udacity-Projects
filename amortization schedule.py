from dateutil.relativedelta import relativedelta
from datetime import datetime

def amortization_calc(principal, annual_rate, period_type, num_periods, loan_date, start_date):
	''' 
	Inputs:
		principal: float. Principal amount
		annual_rate: float. annual simple rate to be charged
		period_type: payment period type. Options 'weekly', 'monthly', 'biweekly'
		num_periods: int, number of periods
		loan_date: string. Date of disbursement of loan. Sample format '1/30/2015'
		start_date: string. Date of first repayment. Sample format '1/30/2015'. Date must be later than loan date.
	Output:
		dictionary: keys = payment dates. values = payment amount.
	'''
	loan_date = datetime.strptime(loan_date, '%m/%d/%Y').date()
	start_date = datetime.strptime(start_date, '%m/%d/%Y').date()
	if period_type == 'weekly':
		period_rate = annual_rate*7./365
		first_period_adjustment = (start_date + relativedelta(days=-7) - loan_date).days*annual_rate*1./365
		assert first_period_adjustment >= 0
		date_list = [start_date + relativedelta(days = 7 * x) for x in range(1,num_periods+1)]
	elif period_type == 'biweekly':
		period_rate = annual_rate*14./365
		first_period_adjustment = (start_date + relativedelta(days=-14) - loan_date).days*annual_rate*1./365
		assert first_period_adjustment >= 0
		date_list = [start_date + relativedelta(days = 14 * x) for x in range(1,num_periods+1)]
	elif period_type == 'monthly':
		period_rate = annual_rate*1./12
		first_period_adjustment = (start_date + relativedelta(months=-1) - loan_date).days*annual_rate*1./365
		assert first_period_adjustment >= 0
		date_list = [start_date + relativedelta(months = x) for x in range(1,num_periods+1)]
	else:
		raise ValueError('period_type argument not understood.') 

	date_list = [datetime.strftime(date, '%m/%d/%Y') for date in date_list]
	payment = (principal * period_rate * (1 + period_rate)**num_periods)/((1+period_rate)**num_periods - 1)
	adjusted_payment = payment + first_period_adjustment/num_periods
	amort_sched = {key: value for (key, value) in zip(date_list, [adjusted_payment]*num_periods)}
	return amort_sched

#dict = amortization_calc(100, 0.1, 'biweekly', 18, '1/1/2015', '2/15/2015')
#print dict

