from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch()
    for tz in ["Asia/Kolkata","UTC","America/New_York"]:
        ctx=b.new_context(timezone_id=tz); pg=ctx.new_page()
        pg.goto("data:text/html,<html></html>")
        r=pg.evaluate("""() => {
          function formatIstClock(date){var utc=date.getTime()+(date.getTimezoneOffset()*60000);
            var ist=new Date(utc+(5.5*3600000));
            return String(ist.getHours()).padStart(2,"0")+":"+String(ist.getMinutes()).padStart(2,"0")+" IST";}
          function formatIstStamp(date){var ist=new Date(date.getTime()+(5.5*3600000));
            var m=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
            return String(ist.getUTCDate()).padStart(2,"0")+" "+m[ist.getUTCMonth()]+" · "
              +String(ist.getUTCHours()).padStart(2,"0")+":"+String(ist.getUTCMinutes()).padStart(2,"0")+" IST";}
          const d=new Date(Date.UTC(2026,8,17,6,25,0));
          return [formatIstClock(d), formatIstStamp(d)];
        }""")
        print(f"{tz:20s} -> clock {r[0]} | stamp {r[1]}   (expect 11:55 IST, 17 SEP)")
        ctx.close()
    b.close()
