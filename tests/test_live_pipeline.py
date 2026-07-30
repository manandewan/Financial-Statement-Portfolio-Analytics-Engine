from src.agents.coordinator import AgentSystemCoordinator

def main():
    print("Testing live agent coordinator pipeline...")
    coordinator = AgentSystemCoordinator()
    bundle = coordinator.run_pipeline(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        risk_free_rate=0.04
    )
    print("Pipeline executed successfully!")
    print(f"Tickers processed: {bundle['tickers']}")
    print("Fundamental metrics extracted:")
    for t, m in bundle['fundamental']['metrics'].items():
        print(f"  {t}: D/E={m.get('debt_to_equity')}, CurrentRatio={m.get('current_ratio')}, ROE={m.get('return_on_equity')}, FCF_Yield={m.get('free_cash_flow_yield')}")
    
    print("\nQuant optimization results:")
    print(f"  Max Sharpe Portfolio: Return={bundle['quant']['max_sharpe_portfolio']['expected_return']:.4f}, Vol={bundle['quant']['max_sharpe_portfolio']['volatility']:.4f}, Sharpe={bundle['quant']['max_sharpe_portfolio']['sharpe_ratio']:.4f}")
    print(f"  Max Sharpe Weights: {bundle['quant']['max_sharpe_portfolio']['weights']}")

if __name__ == "__main__":
    main()
