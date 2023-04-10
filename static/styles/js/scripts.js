(function ($) {
    "use strict";

    /*================================
    Preloader
    ==================================*/

    var preloader = $('#preloader');
    $(window).on('load', function () {
        preloader.fadeOut('slow', function () {
            $(this).remove();
        });
    });

    /*================================
    datatable active
    ==================================*/
    if ($('#dataTable').length) {
        $('#dataTable').DataTable({
            responsive: true
        });
    }
    if ($('#dataTable2').length) {
        $('#dataTable2').DataTable({
            responsive: true
        });
    }
    if ($('#dataTable3').length) {
        $('#dataTable3').DataTable({
            responsive: true
        });
    }



})(jQuery);