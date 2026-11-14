$ErrorActionPreference = 'Continue'
$root = 'C:\Users\yarma\Desktop\Ziwago18'
$ckptBank = 'D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_bankfixed_20261109.pt'
$ckptApical = 'D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_apical2_20261109.pt'
$data = 'D:\Ziwago18-storage\data\hf\squad\language_v19_v6_union\train.jsonl'
$out = 'D:\Ziwago18-storage\Produce'

Set-Location $root
$arms = @(
  @{ name = 'A_bankfixed_apical_off'; ckpt = $ckptBank;   apical = $false },
  @{ name = 'B_apical_apical_on';    ckpt = $ckptApical; apical = $true  },
  @{ name = 'C_apical_apical_off';   ckpt = $ckptApical; apical = $false }
)
foreach ($arm in $arms) {
  $report = Join-Path $out ("v19_compose_" + $arm.name + "_20261109.json")
  $log = Join-Path $out ("v19_compose_" + $arm.name + "_20261109.log")
  $apicalFlag = @()
  if ($arm.apical) { $apicalFlag = @('--apical') }
  Write-Host ("=== ARM " + $arm.name + " ===")
  cmd /c "set PYTHONIOENCODING=utf-8 && python tools/probe_v19_compose.py --checkpoint $($arm.ckpt) --dataset $data --output $report --limit 4 --cpu-threads 6 $($apicalFlag -join ' ') > $log 2>&1"
  Write-Host ("exit=" + $LASTEXITCODE + "  report=" + $report)
}
