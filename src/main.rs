use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, fs::File, io::Read, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let user_message = json!({
        "role": "user",
        "content": args.prompt
    });

    let mut messages = json!([user_message]);

    loop {
        let request = json!({
            "messages": messages,
            "model": "anthropic/claude-haiku-4.5",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "The content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "required": ["command"],
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute"
                            }
                        }
                    }
                }
            }]
        });

        let response: Value = client.chat().create_byot(&request).await?;
        let assistant_message = &response["choices"][0]["message"];

        if let Some(tool_responses) = execute_tool_calls(assistant_message) {
            messages.as_array_mut().unwrap().push(assistant_message.clone());

            // If a tool call was executed, we need to send the tool response back to the model
            for tool_response in tool_responses {
                messages.as_array_mut().unwrap().push(tool_response);
            }
        } else {
            if let Some(content) = assistant_message["content"].as_str() {
                println!("{content}");
            } else {
                eprintln!("No content in response: {response}");
            }
            break;
        }
    }

    Ok(())
}

fn execute_tool_calls(message: &Value) -> Option<Vec<Value>> {
    let tool_calls = &message["tool_calls"];

    if let Some(tool_calls) = tool_calls.as_array() {
        let mut results = Vec::new();
        for call in tool_calls {
            if call["type"] == "function" {
                if let Some(result) = execute_function_call(&call["function"]) {
                    let function_response = json!({
                        "role": "tool",
                        "tool_call_id": call["id"].as_str().unwrap_or_default(),
                        "content": result
                    });
                    results.push(function_response);
                } else {
                    eprintln!("Function call failed: {}", call);
                }
            } else {
                eprintln!("Unexpected tool call: {}", call);
            }
        }

        return Some(results);
    }

    None
}

fn execute_function_call(function: &Value) -> Option<String> {
    let function_name = function["name"].as_str().unwrap_or_default();
    let args = function["arguments"].as_str().unwrap_or("{}");
    let args: serde_json::Value = serde_json::from_str(args).unwrap_or_else(|_| {
        eprintln!("Failed to parse function arguments");
        json!({})
    });

    match function_name {
        "Read" => {
            if let Some(file_path) = args["file_path"].as_str() {
                return read_file(file_path)
                    .map_err(|e| {
                        eprintln!("Failed to read file: {e}");
                        e
                    })
                    .ok();
            } else {
                eprintln!("file_path argument is missing or not a string");
            }
        },
        "Write" => {
            if let Some(file_path) = args["file_path"].as_str() {
                if let Some(content) = args["content"].as_str() {
                    return std::fs::write(file_path, content)
                        .map_err(|e| {
                            eprintln!("Failed to write file: {e}");
                            e
                        })
                        .ok()
                        .map(|_| content.to_string());
                } else {
                    eprintln!("content argument is missing or not a string");
                }
            } else {
                eprintln!("file_path argument is missing or not a string");
            }
        },
        "Bash" => {
            if let Some(command) = args["command"].as_str() {
                return std::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .output()
                    .map_err(|e| {
                        eprintln!("Failed to execute command: {e}");
                        e
                    })
                    .ok()
                    .and_then(|output| {
                        if output.status.success() {
                            String::from_utf8(output.stdout).map_err(|e| {
                                eprintln!("Failed to parse command output: {e}");
                                e
                            }).ok()
                        } else {
                            eprintln!("Command failed with status: {}", output.status);
                            None
                        }
                    });
            } else {
                eprintln!("command argument is missing or not a string");
            }
        },
        _ => eprintln!("Unknown function: {function_name}"),
    }

    None
}

fn read_file(file_path: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}
