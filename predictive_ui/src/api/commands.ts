export async function sendEagleEyeCommand(
  sessionId: string,
  targetAgentId: string,
  question: string
): Promise<{ status: string; message: string }> {
  const payload = {
    action: "interview",
    target_agent_id: targetAgentId,
    questions: [question],
  };

  const response = await fetch(`/api/simulation/${sessionId}/command`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to send command to the simulation engine.");
  }

  return await response.json();
}

export async function sendInterventionCommand(
  sessionId: string,
  event: string
): Promise<{ status: string; message: string }> {
  const payload = {
    action: "intervention",
    event: event,
  };

  const response = await fetch(`/api/simulation/${sessionId}/command`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to send intervention command.");
  }

  return await response.json();
}
