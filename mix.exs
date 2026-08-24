defmodule AxonOnnx.MixProject do
  use Mix.Project

  @source_url "https://github.com/weftspun/nx_shuttle"
  @version "0.1.0"

  def project do
    [
      app: :nx_shuttle,
      version: @version,
      name: "NxShuttle",
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),
      description:
        "Compile Nx defn to an ONNX graph, for accelerator toolchains that consume one",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ],
      aliases: aliases()
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:protox, "~> 1.6.10"},
      {:nx, "~> 0.5", nx_opts()},
      {:pythonx, "~> 0.4.10", only: :test},
      {:jason, "~> 1.2", only: :test},
      {:ex_doc, "~> 0.23", only: :docs}
    ] ++ exla_dep()
  end

  # XLA publishes no windows-x86_64 archive, so EXLA cannot build there at all -- still true,
  # measured on xla v0.10.0 (2026-02-10), which ships nine assets and none for Windows. The
  # reference backend is Nx.BinaryBackend, which needs neither, so this exclusion no longer
  # strands any platform.
  defp exla_dep do
    case :os.type() do
      {:win32, _} ->
        []

      _ ->
        [{:exla, "~> 0.5", [only: :test] ++ exla_opts()}]
    end
  end

  defp package do
    [
      maintainers: ["weftspun"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "NxShuttle",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end

  defp nx_opts do
    if path = System.get_env("NX_ONNX_NX_PATH") do
      [path: path, override: true]
    else
      []
    end
  end

  defp exla_opts do
    if path = System.get_env("AXON_EXLA_PATH") do
      [path: path]
    else
      []
    end
  end

  defp aliases() do
    [
      generate_protobuf:
        "protox.generate --generate-defs-funs=false --keep-unknown-fields=false --multiple-files --output-path=./lib/onnx ./scripts/onnx.proto"
    ]
  end
end
