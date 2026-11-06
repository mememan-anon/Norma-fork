package tool

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestConfiguredUserAgent(t *testing.T) {
	const want = "Lame-Duck/1.0"
	got := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.UserAgent()
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "/final", http.StatusFound)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html><body>ok</body></html>"))
	}))
	defer srv.Close()

	fetch := NewWebFetch(WebFetchConfig{UserAgent: want})
	input, _ := json.Marshal(map[string]string{"url": srv.URL + "/redirect"})
	if _, err := fetch.Call(context.Background(), input, &ToolContext{}); err != nil || got != want {
		t.Fatalf("WebFetch User-Agent = %q, error = %v", got, err)
	}
	got = ""
	fetch = NewWebFetch(WebFetchConfig{UserAgent: want, Proxy: "http://127.0.0.1:1"})
	if _, err := fetch.Call(context.Background(), input, &ToolContext{}); err != nil || got != want {
		t.Fatalf("WebFetch retry User-Agent = %q, error = %v", got, err)
	}

	provider, err := newSearchProvider(WebSearchConfig{Backend: "ddgs", UserAgent: want})
	if err != nil {
		t.Fatal(err)
	}
	provider.(*ddgsProvider).endpoint = srv.URL
	got = ""
	if _, err := provider.Search(context.Background(), "test", 1); err != nil || got != want {
		t.Fatalf("WebSearch User-Agent = %q, error = %v", got, err)
	}

	got = ""
	if _, err := NewWebFetch(WebFetchConfig{}).Call(context.Background(), input, &ToolContext{}); err != nil || got != "norma/0.4" {
		t.Fatalf("default WebFetch User-Agent = %q, error = %v", got, err)
	}
}
