def simulate_policy_gradient(update_fn, filename, init=[0.0]*3):
    a = widgets.FloatSlider(min=0.0, max=1.0, value=1.0, step=0.010)
    b = widgets.FloatSlider(min=0.0, max=1.0, value=1.0, step=0.005)
    c = widgets.FloatSlider(min=0.0, max=1.0, value=1.0, step=0.0025)
    sliders = [a, b, c]
    for i, s in zip(init, sliders):
        s.logit = i
        s.q_val = s.step * 1000
    
    def update_values():
        exps = [np.exp(e.logit) for e in sliders]
        for ex, slid in zip(exps, sliders):
            slid.value = ex / np.sum(exps)
            slid.grad = slid.value * (1-slid.value)
            
    update_values()
    
    def f(a_val10, b_val8, c_val7):
        pass
    
    np.random.seed(1)
    it = 0
    animation_data = []
    while all([v.value < 0.95 for v in [a, b, c]]):
        incr_idx, incr_size = update_fn(sliders)
        update_values()
        it += 1

    print('Done in %s iterations' % it)