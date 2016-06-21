import qt_data_prep as dp
import qt_data_service as ds
import pandas as pd
from global_params import provider
import numpy as np

def alpha_leverage(comp_id,inv_id):
	'''
		This function calculates the alpha capital leverage for a composite, defined as:
		1/(1-beta capital) where beta capital = Leverage x sum(F(i)xBeta(i,j)xcap-rate(j)) 
		Inputs:
			composite id and investment id
		Output:
			DataFrame with columns = composite id and index = time series. 
			In case of research managers, index of preferred gross returns is used.
			cap-rate(j) = beta-loss(j) + margin (j)
	'''
	research_managers = ds.is_research_manager(comp_id)
	if type(comp_id) == int:
		research_managers = pd.Series(index = [comp_id], data = research_managers)
	inv_managers = research_managers[research_managers == 0].index.tolist()
	research_mgrs = research_managers[research_managers == 1].index.tolist()
	weights = ds.get_benchmark_weights(inv_managers,inv_id,'mgr_net',use_active_bmk=False,use_risky_bmk=False,bal_lev_data=None,bmk_weights=None,drop_libor=False,research_manager=False, rtn_idx_time=None)
	del weights['CompositeId']
	weights = weights.reset_index().drop('Date', axis = 1)
	beta_equiv = dp.compute_beta_equivalent(weights)
	unlev_beta_capital = dp.compute_beta_capital(beta_equiv)
	unlev_beta_margins = dp.compute_beta_margins(beta_equiv)
	comb_mgr_rtns = ds.get_manager_returns(comp_id, 'preferred', 'mgr_gross', fund_agg = inv_id, return_calc_rule = False)
	L = get_leverage(comp_id, inv_id,research_managers,comb_mgr_rtns  = comb_mgr_rtns)
	beta_capital = (unlev_beta_capital + unlev_beta_margins)* L
	alpha_capital_leverage = 1/(1- beta_capital)
	return alpha_capital_leverage

def get_leverage(comp_id, inv_id, research_managers, comb_mgr_rtns = None, comb_me = None, comb_mioe = None):
	'''
	This function calculates the leverage for composites based on pre-calced ME and MIOE. For research composites, it gets TGLV.
	Input: 
		comp_id: a single composite id or list of composite. If list, inv_id must be -1
		inv_id: investment id for which leverage is required
		research_managers: Series with index = comp_id and values = 1 for research manager, else 0
	Optional:
		comb_mgr_rtns: DF for manager returns values. index = time series, columns = composite id. This index will given to returned dataframe
	    comb_me: DF for Manager Equity values. index = time series, columns = composite id. If comb_mioe is provided, comb_me must be provided 
		comb_mioe: DF for MIO Equity values. index = time series, columns = composite id. If comb_me is provided, comb_mioe must be provided
	Output:
		leverage: time series index DataFrame with column names = composite id and values = leverage over time
	'''
	
	if type(comp_id) != int and inv_id != -1:
		raise DataException('get_leverage function must be called with inv_id -1 if comp id is not an integer. Passed comp_id = {kwarg}, inv_id = %d'.format(kwarg = comp_id) %(inv_id))
	comp_id_inv = research_managers[research_managers == 0.].index.tolist()
	if len(comp_id_inv) > 0:
		acct_data = provider.get_accounting_data(composite_id = comp_id_inv, cols=['aum'], pre_calced = True)
		acct_data.index.rename(['CompositeId', 'InvestmentId', 'Date'],inplace = True)
		try:
			values_me = acct_data[acct_data['calc_type'] == 'acct_data_me']
			del values_me['calc_type']
			comb_me = values_me.xs(inv_id, level = 'InvestmentId')
			comb_me = comb_me.reset_index()
			comb_me = comb_me.pivot(index = 'Date', columns = 'CompositeId',values = 'aum')
	
			values_mioe = acct_data[acct_data['calc_type'] == 'acct_data_mioe']
			del values_mioe['calc_type']
			comb_mioe = values_mioe.xs(inv_id, level = 'InvestmentId')
			comb_mioe = comb_mioe.reset_index()
			comb_mioe = comb_mioe.pivot(index = 'Date', columns = 'CompositeId',values = 'aum')
		except:
			raise DataException('Didn\'t find pre-calced MIOE or ME for comp_id = {kwarg}, inv_id = %d'.format(kwarg = comp_id) %(inv_id))
		comb_l = comb_me/comb_mioe
		comb_l = (comb_l * (comb_mioe >= 100000)).replace(0, np.NaN)
	else:
		comb_l = pd.DataFrame()
	comb_l = dp.get_research_leverage(research_managers, comb_l, comb_mgr_rtns)
	return comb_l


a = alpha_leverage(919,-1)
print a