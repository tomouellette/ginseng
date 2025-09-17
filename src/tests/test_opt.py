import jax
import jax.numpy as jnp

from ginseng.opt import opt_init_adam, opt_adam_update


def linear() -> dict:
    beta = jnp.array([0.5])
    intercept = jnp.array([0.5])
    return {"beta": beta, "intercept": intercept}


def loss_fn(params, x, y):
    yhat = params["beta"] * x + params["intercept"]
    return ((y - yhat) ** 2).mean()


def test_opt_init_adam():
    opt_init_adam(linear())


def test_opt_adam_update():
    x = jnp.vstack([jnp.linspace(-10, 10) for i in range(10)])
    y = jnp.vstack([jnp.linspace(-10, 10) for i in range(10)])

    params = linear()
    opt_state = opt_init_adam(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params, new_opt_state = opt_adam_update(
            grads, params, opt_state, lr=1e-2, weight_decay=0.1
        )
        return new_params, new_opt_state, loss

    initial_loss = 0.0
    for step in range(10000):
        params, opt_state, loss = train_step(params, opt_state, x, y)

        if step == 0:
            initial_loss = loss

        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    assert (
        loss < initial_loss - 1
    ), "Failure: Loss did not decrease sufficiently in test_opt_adam_update."

    yhat = params["beta"] * x[0:1] + params["intercept"] * x[0:1]

    assert (abs(yhat.flatten()[0]) - 10) < 0.01
    assert (abs(yhat.flatten()[-1]) - 10) < 0.01
