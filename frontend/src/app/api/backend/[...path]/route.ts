import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL = (
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "https://regbrain.onrender.com"
).replace(/\/+$/, "");

const BACKEND_API_KEY =
  process.env.BACKEND_API_KEY ||
  process.env.API_KEY ||
  "regbrain-dev-key";

async function proxyRequest(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> | { path: string[] } }
) {
  const resolvedParams = context.params instanceof Promise ? await context.params : context.params;
  const path = resolvedParams?.path || [];
  const targetPath = Array.isArray(path) ? path.join("/") : path;
  const search = request.nextUrl.search;
  const targetUrl = `${BACKEND_URL}/${targetPath}${search}`;

  const headers: Record<string, string> = {
    "X-API-Key": BACKEND_API_KEY,
    Accept: request.headers.get("accept") || "application/json",
  };

  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers["Content-Type"] = contentType;
  }

  let body: BodyInit | undefined = undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    try {
      body = await request.text();
    } catch {
      // Body may be empty
    }
  }

  try {
    const upstreamRes = await fetch(targetUrl, {
      method: request.method,
      headers,
      body,
      // @ts-expect-error duplex required for streaming in some node fetch versions
      duplex: "half",
    });

    const upstreamContentType = upstreamRes.headers.get("content-type") || "";

    // Handle SSE streams
    if (upstreamContentType.includes("text/event-stream")) {
      return new NextResponse(upstreamRes.body, {
        status: upstreamRes.status,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache, no-transform",
          Connection: "keep-alive",
          "X-Accel-Buffering": "no",
        },
      });
    }

    const resData = await upstreamRes.text();
    return new NextResponse(resData, {
      status: upstreamRes.status,
      headers: {
        "Content-Type": upstreamContentType || "application/json",
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        error: "proxy_error",
        message: `Failed to proxy request to backend: ${error?.message || "Unknown error"}`,
      },
      { status: 502 }
    );
  }
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> | { path: string[] } }
) {
  return proxyRequest(request, context);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> | { path: string[] } }
) {
  return proxyRequest(request, context);
}
