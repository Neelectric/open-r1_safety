def run_protected_adam(
    num_epochs=2,
    lr=5e-5,
    alpha_geom=1.0,
    beta_geom=10.0,
    gamma_exp=0.5,
    subset_update_every=5,
    rho_all=0.99,
    rho_sub=0.99,
):
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    model = copy.deepcopy(base_model).to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]

    state = {}
    for p in params:
        state[p] = {
            "m": torch.zeros_like(p.data),
            "v_all": torch.zeros_like(p.data),
            "v_sub": torch.zeros_like(p.data),
        }

    beta1 = 0.9
    eps = 1e-6
    global_step = 0

    def protected_adam_step():
        nonlocal global_step
        global_step += 1
        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[p]

            # first moment
            s["m"].mul_(beta1).add_(grad, alpha=1 - beta1)

            # second moment on "all" (new English) data
            s["v_all"].mul_(rho_all).addcmul_(grad, grad, value=1 - rho_all)

            v_all = s["v_all"]
            v_sub = s["v_sub"]
            v_protect = alpha_geom * v_all + beta_geom * v_sub

            m_hat = s["m"] / (1 - beta1**global_step)
            denom = (v_protect + eps).pow(gamma_exp)
            step = m_hat / denom
            p.data.add_(step, alpha=-lr)

    def update_subset_curvature():
        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.data
            s = state[p]
            s["v_sub"].mul_(rho_sub).addcmul_(grad, grad, value=1 - rho_sub)

    fr_iter = iter(fr_loader)

    print("=== ProtectedAdam-γ: geometry shaped by French subset ===")
    print(f"alpha_geom={alpha_geom}, beta_geom={beta_geom}, gamma_exp={gamma_exp}")
    print(f"Eval before training:")
    eng_ppl = eval_ppl(model, english_test_ds, "English new")
    fr_ppl  = eval_ppl(model, french_test_ds,  "French protected")
    for epoch in range(num_epochs):
        for step, batch in enumerate(eng_loader):
            # 1) English batch: gradient for new task
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()

            # 2) Take ProtectedAdam step (updates v_all + params)
            protected_adam_step()

            # 3) Occasionally update subset curvature using French
            if (step + 1) % subset_update_every == 0:
                try:
                    fr_batch = next(fr_iter)
                except StopIteration:
                    fr_iter = iter(fr_loader)
                    fr_batch = next(fr_iter)
                fr_batch = {k: v.to(device) for k, v in fr_batch.items()}
                model.zero_grad()
                fr_out = model(**fr_batch)
                fr_loss = fr_out.loss
                fr_loss.backward()
                update_subset_curvature()
                model.zero_grad()

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch} Step {step+1}] loss_new = {loss.item():.4f}")

        print(f"Epoch {epoch} evaluation:")
        eng_ppl = eval_ppl(model, english_test_ds, "English new (ProtectedAdam-γ)")
        fr_ppl  = eval_ppl(model, french_test_ds,  "French protected (ProtectedAdam-γ)")

    return model

protected_model = run_protected_adam(
    num_epochs=3,
    lr=1e-5,
    alpha_geom=1.0,
    beta_geom=10.0,   # strength of protected geometry
    gamma_exp=0.5,    # between 0.5 (Adam) and 1.0 (diag NGD)
    subset_update_every=5,
)
