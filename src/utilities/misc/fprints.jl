# functions for printing and immediately flusing output; these are useful
# for monitoring script progress on a cluster where output is written to
# a log file rather than to terminal.

function fprint(s)
    print(s)
    flush(stdout)
end

function fprintln(s)
    println(s)
    flush(stdout)
end