# Matar todo lo que use el puerto 7860
$pid = (netstat -ano | findstr :7860 | Select-String "LISTENING") -replace '.*\s+(\d+)$','$1'
if ($pid) { taskkill /PID $pid /F }
