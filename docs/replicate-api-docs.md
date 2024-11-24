# Authentication

Whenever you make an API request, you need to authenticate using a token. A token is like a password that uniquely identifies your account and grants you access.

The following examples all expect your Replicate access token to be available from the command line. Because tokens are secrets, they should not be in your code. They should instead be stored in environment variables. Replicate clients look for the `REPLICATE_API_TOKEN` environment variable and use it if available.

To set this up you can use:

```bash
export REPLICATE_API_TOKEN=r8_54y**********************************
```

Some application frameworks and tools also support a text file named `.env` which you can edit to include the same token:

```plaintext
REPLICATE_API_TOKEN=r8_54y**********************************
```

The Replicate API uses the Authorization HTTP header to authenticate requests. If you're using a client library this is handled for you.

You can test that your access token is setup correctly by using our account.get endpoint:

```bash
# What is cURL?
curl https://api.replicate.com/v1/account -H "Authorization: Bearer $REPLICATE_API_TOKEN"
# {"type":"user","username":"aron","name":"Aron Carroll","github_url":"https://github.com/aron"}
```

If it is working correctly you will see a JSON object returned containing some information about your account, otherwise ensure that your token is available:

```bash
echo "$REPLICATE_API_TOKEN"
# "r8_xyz"
```

# Setup

First you'll need to ensure you have a Python environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install the replicate Python library:

```bash
pip install replicate
```

In a `main.py` file, import replicate:

```python
import replicate
```

This will use the `REPLICATE_API_TOKEN` API token you've set up in your environment for authorization.

# Run the model

Use the `replicate.run()` method to run the model:

```python
input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

output = replicate.run(
    "meta/meta-llama-3-8b-instruct",
    input=input
)
print("".join(output))
#=> "The number of parameters in a neural network can impact ...
```

You can learn about pricing for this model on the model page.

The `run()` function returns the output directly, which you can then use or pass as the input to another model. If you want to access the full prediction object (not just the output), use the `replicate.predictions.create()` method instead. This will return a Prediction object that includes the prediction id, status, logs, etc.

# Streaming

This model supports streaming. This allows you to receive output as the model is running:

```python
input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

for event in replicate.stream(
    "meta/meta-llama-3-8b-instruct",
    input=input
):
    print(event, end="")
    #=> "The"
```

## Streaming in the browser

The Python library is intended to be run on the server. However once the prediction has been created its output can be streamed directly from the browser.

The streaming URL uses a standard format called Server Sent Events (or `text/event-stream`) built into all web browsers.

A common implementation is to use a web server to create the prediction using `replicate.predictions.create`, passing the `stream` property set to `true`. Then the `urls.stream` property of the response contains a url that can be returned to your frontend application:

```python
input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

# POST /run_prediction
def handler(request):
    prediction = replicate.predictions.create(
        model="meta/meta-llama-3-8b-instruct",
        input=input,
        stream=True,
    )

    return Response.json({
        'url': prediction.urls['stream']
    })
    # Returns {"url": "https://replicate-stream..."}
```

Make a request to the server to create the prediction and use the built-in `EventSource` object to read the returned url:

```javascript
const response = await fetch("/run_prediction", { method: "POST" });
const { url } = await response.json();

const source = new EventSource(url);
source.addEventListener("output", (evt) => {
  console.log(evt.data) //=> "The"
});
source.addEventListener("done", (evt) => {
  console.log("stream is complete");
});
```

# Prediction lifecycle

Running predictions and trainings can often take significant time to complete, beyond what is reasonable for an HTTP request/response.

When you run a model on Replicate, the prediction is created with a "starting" state, then instantly returned. This will then move to "processing" and eventual one of "successful", "failed" or "canceled".

- Starting
- Running
- Succeeded
- Failed
- Canceled

You can explore the prediction lifecycle by using the `prediction.reload()` method update the prediction to it's latest state.

# Webhooks

Webhooks provide real-time updates about your prediction. Specify an endpoint when you create a prediction, and Replicate will send HTTP POST requests to that URL when the prediction is created, updated, and finished.

It is possible to provide a URL to the `predictions.create()` function that will be requested by Replicate when the prediction status changes. This is an alternative to polling.

To receive webhooks you'll need a web server. The following example uses AIOHTTP, a basic webserver built on top of Python's asyncio library, but this pattern will apply to most frameworks.

Then create the prediction passing in the webhook URL and specify which events you want to receive out of "start", "output" "logs" and "completed":

```python
input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

callback_url = "https://my.app/webhooks/replicate"
replicate.predictions.create(
  model="meta/meta-llama-3-8b-instruct",
  input=input,
  webhook=callback_url,
  webhook_events_filter=["completed"]
)

# The server will now handle the event and log:
#=> Prediction(id='z3wbih3bs64of7lmykbk7tsdf4', ...)
```

The `replicate.run()` method is not used here. Because we're using webhooks, and we don't need to poll for updates.

From a security perspective it is also possible to verify that the webhook came from Replicate, check out our documentation on verifying webhooks for more information.

# Access a prediction

You may wish to access the prediction object. In these cases it's easier to use the `replicate.predictions.create()` function, which return the prediction object.

Though note that these functions will only return the created prediction, and it will not wait for that prediction to be completed before returning. Use `replicate.predictions.get()` to fetch the latest prediction.

```python
import replicate

input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

prediction = replicate.predictions.create(
  model="meta/meta-llama-3-8b-instruct",
  input=input
)
#=> Prediction(id='z3wbih3bs64of7lmykbk7tsdf4', ...)
```

# Cancel a prediction

You may need to cancel a prediction. Perhaps the user has navigated away from the browser or canceled your application. To prevent unnecessary work and reduce runtime costs you can use `prediction.cancel()` method to call the predictions.cancel endpoint.

```python
input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

prediction = replicate.predictions.create(
  model="meta/meta-llama-3-8b-instruct",
  input=input
)

prediction.cancel()
```

# Async Python methods

`asyncio` is a module built into Python's standard library for writing concurrent code using the async/await syntax.

Replicate's Python client has support for asyncio. Each of the methods has an async equivalent prefixed with `async_<name>`.

```python
input = {
    "prompt": "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?",
    "max_new_tokens": 512,
    "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}

prediction = replicate.predictions.create(
  model="meta/meta-llama-3-8b-instruct",
  input=input
)

prediction = await replicate.predictions.async_create(
  model="meta/meta-llama-3-8b-instruct",
  input=input
)
```
# API Input Schema Documentation

## Required Fields

- `prompt` (string)
  - Description: Prompt to send to the model.
  - Order: 0

## Optional Fields

### Core Parameters

- `system_prompt` (string)
  - Description: System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.
  - Default: "You are a helpful assistant"
  - Order: 1

- `max_tokens` (integer)
  - Description: Maximum number of tokens to generate. A word is generally 2-3 tokens.
  - Default: 512
  - Minimum: 1
  - Order: 2

- `min_tokens` (integer)
  - Description: Minimum number of tokens to generate. To disable, set to -1.
  - Minimum: -1
  - Order: 3

### Generation Controls

- `temperature` (number)
  - Description: Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.
  - Default: 0.7
  - Range: 0 to 5
  - Order: 4

- `top_p` (number)
  - Description: When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.
  - Default: 0.95
  - Range: 0 to 1
  - Order: 5

- `top_k` (integer)
  - Description: When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.
  - Default: 0
  - Minimum: -1
  - Order: 6

### Output Controls

- `stop_sequences` (string)
  - Description: A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.
  - Default: "<|end_of_text|>,<|eot_id|>"
  - Order: 7

- `length_penalty` (number)
  - Description: A parameter that controls how long the outputs are. If < 1, the model will tend to generate shorter outputs, and > 1 will tend to generate longer outputs.
  - Default: 1
  - Range: 0 to 5
  - Order: 8

- `presence_penalty` (number)
  - Description: A parameter that penalizes repeated tokens regardless of the number of appearances. As the value increases, the model will be less likely to repeat tokens in the output.
  - Default: 0
  - Order: 9

### Additional Parameters

- `seed` (integer)
  - Description: Random seed. Leave blank to randomize the seed.
  - Order: 10

- `prompt_template` (string)
  - Description: Template for formatting the prompt. Can be an arbitrary string, but must contain the substring `{prompt}`.
  - Default: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
  - Order: 11

- `log_performance_metrics` (boolean)
  - Description: Toggle for logging performance metrics
  - Default: false
  - Order: 12

### Legacy Parameters

- `max_new_tokens` (integer)
  - Description: This parameter has been renamed to max_tokens. max_new_tokens only exists for backwards compatibility purposes. We recommend you use max_tokens instead. Both may not be specified.
  - Minimum: 1
  - Order: 13

- `min_new_tokens` (integer)
  - Description: This parameter has been renamed to min_tokens. min_new_tokens only exists for backwards compatibility purposes. We recommend you use min_tokens instead. Both may not be specified.
  - Minimum: -1
  - Order: 14

## Output Schema

- Type: array
- Items Type: string
- Title: Output
- Array Type: iterator
- Array Display: concatenate