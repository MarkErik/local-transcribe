# OpenAI harmony response format

The [`gpt-oss` models](https://openai.com/open-models) were trained on the harmony response format for defining conversation structures, generating reasoning output and structuring function calls. If you are not using `gpt-oss` directly but through an API or a provider like Ollama, you will not have to be concerned about this as your inference solution will handle the formatting. If you are building your own inference solution, this guide will walk you through the prompt format. The format is designed to mimic the OpenAI Responses API, so if you have used that API before, this format should hopefully feel familiar to you. `gpt-oss` should not be used without using the harmony format, as it will not work correctly.

## Concepts

### Roles

Every message that the model processes has a role associated with it. The model knows about five types of roles:

| Role        | Purpose                                                                                                                                                                                 |
| :---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `system`    | A system message is used to specify reasoning effort, meta information like knowledge cutoff and built-in tools                                                                         |
| `developer` | The developer message is used to provide information about the instructions for the model (what is normally considered the “system prompt”) and available function tools                |
| `user`      | Typically representing the input to the model                                                                                                                                           |
| `assistant` | Output by the model which can either be a tool call or a message output. The output might also be associated with a particular “channel” identifying what the intent of the message is. |
| `tool`      | Messages representing the output of a tool call. The specific tool name will be used as the role inside a message.                                                                      |

These roles also represent the information hierarchy that the model applies in case there are any instruction conflicts: `system` \> `developer` \> `user` \> `assistant` \> `tool`

#### Channels

Assistant messages can be output in three different “channels”. These are being used to separate between user-facing responses and internal facing messages.

| Channel      | Purpose                                                                                                                                                                                                                                                                                                                                                            |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `final`      | Messages tagged in the final channel are messages intended to be shown to the end-user and represent the responses from the model.                                                                                                                                                                                                                                 |
| `analysis`   | These are messages that are being used by the model for its chain of thought (CoT). **Important:** Messages in the analysis channel do not adhere to the same safety standards as final messages do. Avoid showing these to end-users.                                                                                                                             |
| `commentary` | Any function tool call will typically be triggered on the `commentary` channel while built-in tools will normally be triggered on the `analysis` channel. However, occasionally built-in tools will still be output to `commentary`. Occasionally this channel might also be used by the model to generate a [preamble](#preambles) to calling multiple functions. |


## Prompt format

If you choose to build your own renderer, you’ll need to adhere to the following format.

### Special Tokens

The model uses a set of special tokens to identify the structure of your input. All special tokens follow the format `<|type|>`.

| Special token           | Purpose                                                                                                                                     | Token ID |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------- |
| <&#124;start&#124;>     | Indicates the beginning of a [message](#message-format). Followed by the “header” information of a message starting with the [role](#roles) | `200006` |
| <&#124;end&#124;>       | Indicates the end of a [message](#message-format)                                                                                           | `200007` |
| <&#124;message&#124;>   | Indicates the transition from the message “header” to the actual content                                                                    | `200008` |
| <&#124;channel&#124;>   | Indicates the transition to the [channel](#channels) information of the header                                                              | `200005` |
| <&#124;constrain&#124;> | Indicates the transition to the data type definition in a [tool call](#receiving-tool-calls)                                                | `200003` |
| <&#124;return&#124;>    | Indicates the model is done with sampling the response message. A valid “stop token” indicating that you should stop inference.             | `200002` |
| <&#124;call&#124;>      | Indicates the model wants to call a tool. A valid “stop token” indicating that you should stop inference.                                   | `200012` |

### Message format

The harmony response format consists of “messages” with the model potentially generating multiple messages in one go. The general structure of a message is as follows:

```
<|start|>{header}<|message|>{content}<|end|>
```

The `{header}` contains a series of meta information including the [role](#roles). `<|end|>` represents the end of a fully completed message but the model might also use other stop tokens such as `<|call|>` for tool calling and `<|return|>` to indicate the model is done with the completion.

### Chat conversation format

Following the message format above the most basic chat format consists of a `user` message and the beginning of an `assistant` message.

#### Example input

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant
```

The output will begin by specifying the `channel`. For example `analysis` to output the chain of thought. The model might output multiple messages (primarily chain of thought messages) for which it uses the `<|end|>` token to separate them.

Once its done generating it will stop with either a `<|return|>` token indicating it’s done generating the final answer, or `<|call|>` indicating that a tool call needs to be performed. In either way this indicates that you should stop inference.

#### Example output

```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

The `final` channel will contain the answer to your user’s request. Check out the [reasoning section](#reasoning) for more details on the chain-of-thought.

**Implementation note:** `<|return|>` is a decode-time stop token only. When you add the assistant’s generated reply to conversation history for the next turn, replace the trailing `<|return|>` with `<|end|>` so that stored messages are fully formed as `<|start|>{header}<|message|>{content}<|end|>`. Prior messages in prompts should therefore end with `<|end|>`. For supervised targets/training examples, ending with `<|return|>` is appropriate; for persisted history, normalize to `<|end|>`.

### System message format

The system message is used to provide general information to the system. This is different to what might be considered the “system prompt” in other prompt formats. For that, check out the [developer message format](#developer-message-format).

We use the system message to define:

1. The **identity** of the model — This should always stay as `You are ChatGPT, a large language model trained by OpenAI.` If you want to change the identity of the model, use the instructions in the [developer message](#developer-message-format).
2. Meta **dates** — Specifically the `Knowledge cutoff:` and the `Current date:`
3. The **reasoning effort** — As specified on the levels `high`, `medium`, `low`
4. Available channels — For the best performance this should map to `analysis`, `commentary`, and `final`.

**If you are defining functions,** it should also contain a note that all function tool calls must go to the `commentary` channel.

For the best performance stick to this format as closely as possible.

#### Example system message

The most basic system message you should use is the following:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
```

If functions calls are present in the developer message section, use:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|>
```

### Developer message format

The developer message represents what is commonly considered the “system prompt”. It contains the instructions that are provided to the model.

Your developer message could look like this:

```
<|start|>developer<|message|># Instructions

{instructions}<|end|>
```

Where `{instructions}` is replaced with your “system prompt”.


### Reasoning

The gpt-oss models are reasoning models. By default, the model will do medium level reasoning. To control the reasoning you can specify in the [system message](#system-message-format) the reasoning level as `low`, `medium`, or `high`. The recommended format is:

```
Reasoning: high
```

The model will output its raw chain-of-thought (CoT) as assistant messages into the `analysis` channel while the final response will be output as `final`.

For example for the question `What is 2 + 2?` the model output might look like this:

```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

In this case the CoT is

```
User asks: “What is 2 + 2?” Simple arithmetic. Provide answer.
```

And the actual answer is:

```
2 + 2 = 4
```

**Important:**  
The model has not been trained to the same safety standards in the chain-of-thought as it has for final output. You should not show the chain-of-thought to your users, as they might contain harmful content.

#### Handling reasoning output in subsequent sampling

In general, you should drop any previous CoT content on subsequent sampling if the responses by the assistant ended in a message to the `final` channel. Meaning if our first input was this:

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant
```

and resulted in the output:

```
<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
```

For the model to work properly, the input for the next sampling should be

```
<|start|>user<|message|>What is 2 + 2?<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|end|>
<|start|>user<|message|>What about 9 / 2?<|end|>
<|start|>assistant
```
